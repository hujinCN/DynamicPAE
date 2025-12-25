import functools
import random
from typing import Optional

import PIL.Image
import torch
import torch.nn
import torchvision.transforms
from einops import rearrange
from torchattacks import UPGD
from torchvision.ops import box_convert
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.v2 import Compose
# MOCO memory bank
from ultralytics.utils import ops

from dynamic_example.models import EnvEncoder2, ResVAE2, tv_loss, GeoLoss
from dynamic_example.models.gradient_hacker import GradNorm, GradRescale, skip_grad
from dynamic_example.models.prob_models import TaskRatioSampler, MotionSimulator, TaskWeightBalancing, ColorSimulator
from dynamic_example.utils.transforms import ApplyPatchToBBox, CamTrans, compose_boxes
from dynamic_example.utils.transforms_color import gaussian_blur, ColorTrans


class OptimizationStages:
    DISTORTION = 0
    PATCH_REGULARIZATION_G = 1
    PATCH_REGULARIZATION_D = 2
    VAE = 3
    ATK = 10
    ATK_GRADSCALE = 11
    ATK_STAT_UNNORMED = 12
    DISTORTION_STAT_UNNORMED = 13
    NEXT_STAGE = 999


class Scenario(torch.nn.Module):
    def __init__(self, tensorboard_logger, log_loss=True):
        super().__init__()
        self.logger = tensorboard_logger
        self.do_log_loss = log_loss
        self._cur_name = ""
        self.__tot_losses_stack__ = []

        def empty(grad, stage):
            return grad

        self.backward_controller = empty
        self.emb_logger = lambda *args, **kwargs: None

    @property
    def __tot_losses__(self):
        if self.__tot_losses_stack__.__len__() == 0:
            self.__tot_losses_stack__.append(dict())
        return self.__tot_losses_stack__[-1]

    def push_loss_stack(self):
        for x in self.children():
            if isinstance(x, Scenario):
                x.push_loss_stack()
        self.__tot_losses_stack__.append(dict())

    def pop_loss_stack(self):
        for x in self.children():
            if isinstance(x, Scenario):
                x.pop_loss_stack()
        if self.__tot_losses_stack__.__len__() > 0:
            self.__tot_losses_stack__.pop(-1)

    def set_curriculum_description(self, name):
        self._cur_name = name

    def pre_process_data(self, *args, **kwargs):
        raise NotImplementedError()

    def setup(self, device):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def log_loss(self, loss_name, data):
        if self.do_log_loss:
            # print(f"{loss_name}:  {data}")
            if isinstance(data, torch.Tensor) and len(data.shape) > 0:
                data = data.mean()
            self.logger(self._cur_name + "/" + loss_name, data)

    def fetch_total_loss(self, stage=0, all=False):
        """
        @param stage: OptimizationStages.XX
        @param all: when all == True, mean reduction is performed
        @return: torch.Tensor or 0.0
        """
        if isinstance(stage, list):
            ret = 0.0
            for i in stage:
                ret = ret + self.fetch_total_loss(i)
            return ret
        loss = 0.0
        if all:
            for x in self.__tot_losses__:
                loss = loss + self.__tot_losses__[x].mean()
        elif self.__tot_losses__.__contains__(stage) and self.__tot_losses__[stage] is not None:
            loss = loss + self.__tot_losses__[stage]
            # self.__tot_losses__[stage] = None
        for x in self.children():
            if isinstance(x, Scenario):
                loss = loss + x.fetch_total_loss(stage, all=all)
        return loss

    def reset_losses(self):
        for x in self.children():
            if isinstance(x, Scenario):
                x.reset_losses()
        self.__tot_losses_stack__ = []
        # for k in self.__tot_losses__.keys():
        #     self.__tot_losses__[k] = None

    def add_loss(self, stage, value, deformable=False):
        if not self.training:
            return

        if not self.__tot_losses__.__contains__(stage) or self.__tot_losses__[stage] is None:
            self.__tot_losses__[stage] = value
        else:
            shape_same = isinstance(self.__tot_losses__[stage], (int, float)) or self.__tot_losses__[
                stage].shape == value.shape
            if deformable and not shape_same:
                self.__tot_losses__[stage] += value.sum() / self.__tot_losses__[stage].shape[0]
            else:
                assert shape_same, "Cannot accumulate: Loss shapes are Different."
                self.__tot_losses__[stage] += value

    def attach_controller(self, fn):
        self.backward_controller = fn


class SenarioLosses(torch.nn.Module):
    def __init__(self, parent_scenario: Scenario):
        super().__init__()
        self._parent_scenario = [parent_scenario]

    @property
    def parent_scenario(self):
        return self._parent_scenario[0]


# Affine Medical Image Registration with Coarse-to-Fine Vision Transformer  # Registration

from dynamic_example.tasks.loss_integration import ObjectiveLossesWithGrad, TrainVAE, GANLoss, \
    LossTorchATK


class UAPBaseline(Scenario):
    def __init__(self, logger, detector, hparams, loss_cls=GeoLoss):
        super().__init__(logger)
        self.patch_size = hparams.patch_pixel
        self._adv_example = torch.nn.Parameter(torch.ones(1, 3, self.patch_size, self.patch_size) * 0.5)

        self.patch_applier = ApplyPatchToBBox()
        self.transform_patch = MotionSimulator(**hparams.motion_patch)
        self.objective_loss = ObjectiveLossesWithGrad(self, hparams, loss_cls)
        self.hparams = hparams
        self.model = detector
        self.capture_field_mu = 2.0
        self.input_size = self.hparams.input_pixel
        self.cam_applier = CamTrans(new_shape=(self.hparams.input_pixel, self.hparams.input_pixel))

        self.color_sampler = ColorSimulator()
        self.color_transform = ColorTrans()

    def get_adv_example(self):
        return self._adv_example

    @property
    def num_stages(self):
        return 1

    def get_patch_loc(self, bboxes):
        pp = self.transform_patch.sample(bboxes, sample_around=True)
        pp[:, :2].clip_(min=0.02, max=0.98)
        pp[:, 2:4].clip_(min=0.02, max=1)
        return pp

    def forward(self, imgs, bboxes, format="xywh", val=False, visualize=False, val_single=False):
        assert bboxes.max() <= 2.0, "Please use normalized bbox for input."
        self.visualize_cache = []
        bboxes = box_convert(bboxes, format, "cxcywh")
        patch_loc = self.get_patch_loc(bboxes)
        patch = self.get_adv_example().repeat(imgs.shape[0], 1, 1, 1)

        if self.hparams.gaussian_blur:
            patch = gaussian_blur(patch, [3, 3], torch.ones_like(imgs[:, 0, 0, 0]))
        patch = patch.clip(min=0, max=1)
        patch = self.color_transform(patch,
                                     self.color_sampler.sample(imgs)) if self.color_sampler is not None else patch
        # clip_grad(patch)
        self.patch_applier.update_affine_mat_transform(patch_loc, patch, imgs)
        imgsp = self.patch_applier(imgs, patch)

        if val:
            return [(patch, patch_loc)]
        if val_single:
            return [imgsp]

        self.log_loss("max_patch_value", patch.max())
        self.objective_loss(imgs, imgsp, imgsp, bboxes, "cxcywh", self.model)

        if visualize:
            self.visualize_cache.append(imgs.detach())
            self.visualize_cache.append(patch.detach())
            self.visualize_cache.append(imgsp.detach())


class UAPFile(UAPBaseline):
    def __init__(self, logger, detector, hparams, file):
        super().__init__(logger, detector, hparams)
        img = PIL.Image.open(file)
        tr = Compose([
            Resize(self.patch_size),
            ToTensor()
        ])

        self.adv_p = tr(img).unsqueeze(0)

    def get_adv_example(self):
        return self.adv_p.to(self._adv_example)


class TorchATKScenario(UAPBaseline):

    def __init__(self, logger, detector, hparams, loss_cls=GeoLoss):
        super().__init__(logger, detector, hparams)
        self.torchatk_loss = LossTorchATK(self, hparams, loss_cls)
        self.atk_obj = UPGD(model=self.model, eps=1.0, steps=100, random_start = False)
        self.atk_obj.model.training = False
        self.atk_obj.model.eval = lambda **kwargs: self.atk_obj.model

        self._adv_example = torch.nn.Parameter(torch.ones(self.hparams.batch_size, 3, self.patch_size, self.patch_size) * 0.5)

    @property
    def num_stages(self):
        return 1

    def apply_and_loss(self, img_adv, bboxes, label1, img_orig):
        patch_loc = self.get_patch_loc(bboxes)
        self.patch_applier.update_affine_mat_transform(patch_loc, img_adv.clone(), img_orig)

        img_adv1 = self.color_transform(img_adv,
                                     self.color_sampler.sample(img_orig)) if self.color_sampler is not None else img_adv
        # print(img_adv.max(), img_adv.min())
        img_adv1 = gaussian_blur(img_adv1, [15, 15],  torch.ones_like(img_adv[:, 0, 0, 0]) * self.patch_applier.get_sample_length().mean(dim=-1) / 2)
        imgsp = self.patch_applier(img_orig, img_adv1)
        ls = - self.torchatk_loss(imgsp, bboxes, self.model, "cxcywh").sum() - tv_loss(img_adv).sum() * 1.0 # - reduction((img_adv - 0.5).abs()).sum() * 0.01

        return ls

    def forward(self, imgs, bboxes, format="xywh", val=False, visualize=False, val_single=False, patch = None):
        assert bboxes.max() <= 2.0, "Please use normalized bbox for input."
        self.visualize_cache = []

        if patch is not None:
            patch_loc = self.get_patch_loc(bboxes)

            patch = self.color_transform(patch,
                                         self.color_sampler.sample(patch)) if self.color_sampler is not None else patch
            self.patch_applier.update_affine_mat_transform(patch_loc, patch, imgs)
            imgsp = self.patch_applier(imgs, patch)

            # clip_grad(patch)

            if val:
                return [(patch, patch_loc)]
            if val_single:
                return [imgsp]
            else:
                assert "only support eval mode"


        bboxes = box_convert(bboxes, format, "cxcywh")
        with torch.set_grad_enabled(True): # Torch Lightning Automatic Banned
            with torch.inference_mode(False):
                self.atk_obj.loss = functools.partial(self.apply_and_loss, img_orig=imgs)
                patch = self.atk_obj(self.get_adv_example()[:imgs.shape[0]], bboxes)
        results = []
        for i in range(self.num_stages):
            results = results + self.forward(imgs, bboxes, "cxcywh", val, visualize, val_single, patch)


        # self.log_loss("max_patch_value", patch.max())
        # self.objective_loss(imgs, imgsp, imgsp, bboxes, "cxcywh", self.model)

        if visualize:
            self.visualize_cache.append(imgs.detach())
            self.visualize_cache.append(patch.detach())
            if val_single:
                self.visualize_cache.append(results[0].detach())

        return results


class ScenarioDynamicPretrain(Scenario):
    def __init__(self, logger, detector, generator: EnvEncoder2, generative_nn: ResVAE2, hparams,
                 loss_cls=GeoLoss, ood_detector=None, task_balacing: Optional[TaskWeightBalancing] = None):
        super().__init__(logger)
        self.patch_applier = ApplyPatchToBBox()
        # self.transform_cam = MotionSimulator(v_scale=0.005, v_xy=0.05, force_ratio=True)
        self.transform_patch = MotionSimulator(**hparams.motion_patch)
        self.color_sampler = ColorSimulator()
        self.color_transform = ColorTrans()
        self.objective_loss = ObjectiveLossesWithGrad(self, hparams, loss_cls, ood_detector)
        self.hparams = hparams
        self.detector = detector
        self.generator = generator
        self.capture_field_mu = 2.0
        self.input_size = self.hparams.input_pixel
        self.cam_applier = CamTrans(new_shape=(self.hparams.input_pixel, self.hparams.input_pixel))
        self.task_ratio_sampler = TaskRatioSampler()
        self.grad_norm = GradNorm(num_stages=2)
        self.grad_rescale = GradRescale()
        self.task_balancing: Optional[TaskWeightBalancing] = task_balacing
        self.patch_size = hparams.patch_pixel

        self.generative_nn = generative_nn
        self.loss_generative_nn = TrainVAE(self, generative_nn, ood_detector)

        self.gan_loss = GANLoss(self, hparams, ood_detector) if self.hparams.use_gan else None
        # if self.hparams.lambda_vae_adv != 0.0:
        #     self.styleLoss = StyleLoss()

        self.visualize_cache = []
        self.env_traing_aug = torchvision.transforms.RandAugment() if getattr(self.hparams, "use_randaug", True) else (lambda x: x.clone())
        # self.eval()

    def setup(self, device):
        log_loss = self.do_log_loss
        self.do_log_loss = False
        train = self.training
        self.train()
        self.forward(torch.zeros(4, 3, self.input_size, self.input_size, device=device),
                     torch.tensor([[0.4, 0.4, 0.1, 0.1]] * 4, device=device), training_stage=2)
        self.reset_losses()
        self.train(train)
        self.do_log_loss = log_loss

    @property
    def num_stages(self):
        return 3

    def get_patch_loc(self, bboxes):
        pp = self.transform_patch.sample(bboxes, sample_around=True)
        pp[:, :2].clip_(min=0.02, max=0.98)
        pp[:, 2:4].clip_(min=0.02, max=1)
        return pp

    # def get_patch_loc(self, imgs, bboxes):
    #     bs = imgs.shape[0]
    #     ccwhrrr_patch = self.transform_patch.sample(torch.zeros([bs, 7]).to(imgs))
    #     ccwhrrr_patch[:, 4:] = 0
    #     ccwhrrr_patch[:, :4] = bboxes
    #     return ccwhrrr_patch

    def forward(self, imgs, bboxes, format="xywh", val=False, val_single=False, visualize=False, ratio=None,
                training_stage=0, meta_data = None, meta_header = []):
        if val or val_single:
            training_stage = 3
        assert imgs.shape[-1] == self.input_size
        assert bboxes.max() <= 2.0, f"Please use normalized bbox for input, cur bbox: {bboxes}"

        if meta_data is None:
            meta_data = [[] for _ in range(imgs.shape[0])]
        if (val or val_single) and ratio is None:
            ratio_list = [[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]]
            return self(imgs, bboxes, format, val, val_single, ratio=ratio_list[0], meta_data = [meta_data[j] + [1] for j in range(imgs.shape[0])], meta_header = meta_header +["ratio"]) + \
                self(imgs, bboxes, format, val, val_single, ratio=ratio_list[1], meta_data = [meta_data[j] + [2] for j in range(imgs.shape[0])], meta_header = meta_header +["ratio"]) + \
                self(imgs, bboxes, format, val, val_single, ratio=ratio_list[2], meta_data = [meta_data[j] + [3] for j in range(imgs.shape[0])], meta_header = meta_header +["ratio"])

        bboxes = box_convert(bboxes, format, "cxcywh")

        self.visualize_cache = []
        self.embedding_cache = []
        patch_loc = self.get_patch_loc(bboxes)
        patch = torch.ones(imgs.shape[0], imgs.shape[1], self.patch_size, self.patch_size).to(imgs) * 0.5
        self.task_ratio_sampler.set_val_ratio(ratio)

        mask = None
        if not self.hparams.mask_output:
            mask = None
        elif training_stage <= 2:
            while mask is None or torch.any(mask.sum(dim=-1) == 0):
                mask = torch.randint(2, [imgs.shape[0], 81], device=imgs.device)  # COCO class size
                mask[:, self.hparams.det_loss.supress_clsses] = 0
            # mask[::2, 1:] = 1
        else:
            mask = torch.ones(imgs.shape[0], 81,
                              device=imgs.device)  # 0: attacked class(surpress). 1: other class(magnitude)
            mask[:, self.hparams.det_loss.supress_clsses] = 0

        task_ratio = self.task_ratio_sampler(imgs)
        # task_ratio[:, 0] *= 2

        # clip_grad(patch)
        self.patch_applier.update_affine_mat_transform(patch_loc, patch, imgs)

        img_patch, imgs_aug = self.do_aug(imgs, patch, visualize)

        img_samples = None

        # Dynamic Patch Background: odd: changed.
        if not (val or val_single):
            img_samples = imgs_aug[:, :,
                          (self.input_size // 2 - self.patch_size // 2):(self.input_size // 2 + self.patch_size // 2),
                          (self.input_size // 2 - self.patch_size // 2):(
                                  self.input_size // 2 + self.patch_size // 2)].detach()
            # imgs[1::2] = self.patch_applier(imgs, torch.flip(img_samples.detach(), [0]))[1::2] # use odd image for gen training
            img_samples = img_samples[::2]  # use even image for VAE training

        self.patch_applier.reverse()
        patch_bg = self.patch_applier(patch.detach(), imgs)  # TODO: simplify this
        self.patch_applier.reverse()

        if getattr(self.hparams, "val_resample", False):
            patch_loc = self.get_patch_loc(bboxes)
            self.patch_applier.update_affine_mat_transform(patch_loc, patch, imgs)
        sample_scale = self.patch_applier.sample_length.mean(dim=-1)

        if training_stage == 0:
            with skip_grad(self.generator):
                patch, kl_losses0, kl_losses1 = self.generator(None, task_ratio=task_ratio, patch=patch_bg,
                                                               img_extra_channel=None,
                                                               detach_vae=self.hparams.detach_vae,
                                                               rand_vae=self.hparams.rand_local, mask=mask)
        else:
            patch, kl_losses0, kl_losses1, embeddings = self.generator(imgs_aug if training_stage > 1 else None,
                                                           task_ratio=task_ratio, patch=patch_bg,
                                                           img_extra_channel=img_patch if not hasattr(self.hparams,
                                                                                                      "wopos") else None,
                                                           detach_vae=self.hparams.detach_vae,
                                                           rand_vae=self.hparams.rand_local, mask=mask, return_emb = True)

            # if val or val_single:
            #     self.emb_logger(embeddings, meta_data, meta_header)

        if getattr(self.hparams, "norm_bnd", False):
            patch = patch * (task_ratio[:, 0,None,  None, None]) + patch_bg * (1 - task_ratio[:, 0,None,  None, None])
            task_ratio = task_ratio.clone()
            task_ratio[:, 0] = 1
            task_ratio[:, 1] = 0




        if self.hparams.gaussian_blur:
            patch_gb = gaussian_blur(patch, [15, 15], torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)
        else:
            patch_gb = patch
        if val or val_single:
            patch_gb = torch.nan_to_num(patch_gb, nan=0.0, posinf=0.0, neginf=0.0)
            patch_val = self.color_transform(patch_gb, self.color_sampler.sample(
                imgs)) if self.color_sampler is not None else patch_gb

            if val:
                return [(patch_val, patch_loc)]

            if val_single:
                imgs_patch = self.patch_applier(imgs, patch_val)
                return [imgs_patch]

        if hasattr(self.hparams, "center_rebalance"):
            ratiocenter = getattr(self.hparams, "center_rebalance")
            task_ratio = task_ratio.clone()
            kl_losses0 = kl_losses0.clone()
            task_ratio[self.task_ratio_sampler.get_atk_batches(task_ratio), 0] *= ratiocenter
            kl_losses0[self.task_ratio_sampler.get_atk_batches(task_ratio)] *= ratiocenter

            # self.add_loss(OptimizationStages.DISTORTION, kl_losses0)
        if self.hparams.lambda_vae != 0.0:
            img_samples = torch.cat([img_samples, patch_bg[::2]],
                                    dim=0)  # use even patch background for balanced VAE training

            if getattr(self.hparams, "extra_samples", False):
                img_samples = patch_bg
            if training_stage > 0 and self.hparams.vae_reinclude_sample:
                img_samples = torch.cat([img_samples, patch.detach()[patch.shape[0] // 4:]], dim=0)
            self.loss_generative_nn(
                img_samples, "gen", self.hparams.lambda_vae)

        # if self.hparams.lambda_vae_adv != 0.0:
        #     if self.gan_loss is not None and training_stage == 2:  # use GAN loss
        #         self.gan_loss.forward(img_samples, patch[task_ratio[:, 0] > task_ratio[:, 1]],
        #                               self.hparams.lambda_vae_adv)
        #     else:
        #         block = patch.shape[0] // 4
        #         if training_stage == 1:
        #             self.loss_generative_nn.forward_adv(patch_bg, patch.detach(), self.hparams.lambda_vae_adv)
        #         if training_stage == 2:
        #             self.loss_generative_nn.forward_adv(torch.cat([patch_bg[block:], patch[block: 2 * block]], dim=0),
        #                                                 patch[block * 2:], self.hparams.lambda_vae_adv)

        # self.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, kl_losses1)
        if isinstance(kl_losses0, torch.Tensor):
            self.log_loss("kl_loss_env", kl_losses0.mean())
        if isinstance(kl_losses1, torch.Tensor):
            self.log_loss("kl_loss_p", kl_losses1.mean())
        if isinstance(kl_losses0, torch.Tensor) and not (getattr(self.hparams, "abl_kl_constrain", False) and training_stage > 1):
            self.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, kl_losses0 * getattr(self.hparams, "kl_constrain", self.hparams.lambda_vae) / 2,
                          deformable=True)
        if isinstance(kl_losses1, torch.Tensor) and not (getattr(self.hparams, "abl_kl_constrain", False) and training_stage > 1):
            self.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, kl_losses1 * getattr(self.hparams, "kl_constrain", self.hparams.lambda_vae) / 2,
                          deformable=True)

        # if self.hparams.grad_task_norm:

        ratio = 1 if self.task_balancing is None else self.task_balancing.ratio[0].detach()
        tw = torch.tensor([self.hparams.lambda_atk, self.hparams.lambda_distortion * ratio]).to(imgs)
        # patch = self.grad_norm(patch, task_ratio * tw)
        # else:
        #     patch_grad_norm = patch

        #        + gaussian blur -> patch_gb -> magnitude loss,  + color jitter -> validation
        # patch -> tv loss
        #        + grad_rescale -> patch_grad_rescale, -> +color jitter + gaussian blur -> attack loss
        #
        if self.hparams.lambda_tv != 0.0:
            loss_tv = tv_loss(patch)
            self.log_loss("atk_loss_tv", loss_tv)
            self.add_loss(OptimizationStages.DISTORTION, loss_tv * self.hparams.lambda_tv)
        # if self.hparams.lambda_vae_adv != 0.0 and training_stage >= 1:
        #     style_loss = self.styleLoss(imgs, torch.nn.AvgPool2d(kernel_size=4, stride=4)(patch)) * 100
        #     style_loss[:imgs.shape[0]//2] = 0
        #     self.log_loss("loss_style", style_loss)
        #     self.add_loss(OptimizationStages.DISTORTION, style_loss * self.hparams.lambda_vae_adv)

        if self.hparams.grad_rescale:
            patch_grad_rescale = self.grad_rescale(patch)
        else:
            patch_grad_rescale = patch

        if self.hparams.gaussian_blur:
            patch_grad_rescale = gaussian_blur(patch_grad_rescale, [15, 15],
                                               torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)

        if self.color_sampler is not None:
            patch_grad_rescale = self.color_transform(patch_grad_rescale, self.color_sampler.sample(imgs))
        # self.add_loss(OptimizationStages.DISTORTION, loss_tv * self.hparams.lambda_tv)
        # self.log_loss("atk_loss_tv", loss_tv)
        # self.add_loss(OptimizationStages.DISTORTION, loss_tv * self.hparams.lambda_tv)

        # print(self.patch_applier.sample_length)

        imgs_patch = self.patch_applier(imgs, patch_gb)
        imgs_patch_gradscale = self.patch_applier(imgs, patch_grad_rescale)
        self.objective_loss(imgs, imgs_patch, imgs_patch_gradscale, bboxes, "cxcywh", self.detector, mask=mask,
                            sample_size=sample_scale ** 2, inv_scale=task_ratio[:, 1] * tw[1],
                            atk_scale=task_ratio[:, 0] * tw[0], eval_mode=training_stage > 2)


        if visualize:
            self.visualize_cache.append(imgs.detach())
            self.visualize_cache.append(patch.detach())
            self.visualize_cache.append(imgs_patch_gradscale.detach())

    def do_aug(self, imgs, patch, visualize):
        img_patch = self.patch_applier(torch.zeros_like(imgs), patch)
        if visualize:
            self.visualize_cache.append(img_patch.clone())
        if self.training:
            imgs_aug = torch.zeros_like(imgs)
            for i in range(imgs.shape[0]):
                img_proc = (torch.stack([imgs[i], img_patch[i]], dim=0) * 255).to(torch.uint8)
                proc = False
                img_proc1 = None
                while not proc or int((img_proc1[1] > 1).sum()) < 5 * 3:  # 5 pix
                    img_proc1 = self.env_traing_aug(img_proc)
                    proc = True
                img_proc1 = img_proc1.to(imgs) / 255
                imgs_aug[i] = img_proc1[0]
                img_patch[i] = img_proc1[1]

        else:
            imgs_aug = imgs
        img_patch = (img_patch * 4).clip(max=1.0)
        img_patch = img_patch.mean(dim=1, keepdim=True)
        if visualize:
            self.visualize_cache.append(imgs_aug.detach())
            self.visualize_cache.append(img_patch.detach())
        return img_patch, imgs_aug


class ScenarioFinetuneUAP(ScenarioDynamicPretrain):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.param_atk = torch.nn.Parameter(torch.zeros(self.generative_nn.latent_dim))

    @property
    def num_stages(self):
        return 1

    def forward(self, imgs, bboxes, format="xywh", val=False, val_single=False, visualize=False, ratio=None,
                training_stage=0):
        if val or val_single:
            training_stage = 3
        assert imgs.shape[-1] == self.input_size
        assert bboxes.max() <= 2.0, f"Please use normalized bbox for input, cur bbox: {bboxes}"
        ratio = [1.0, 0.0]

        bboxes = box_convert(bboxes, format, "cxcywh")

        self.visualize_cache = []
        patch_loc = self.get_patch_loc(bboxes)
        patch = torch.ones(imgs.shape[0], imgs.shape[1], self.patch_size, self.patch_size).to(imgs) * 0.5

        self.task_ratio_sampler.set_val_ratio(ratio)

        self.patch_applier.update_affine_mat_transform(patch_loc, patch, imgs)

        sample_scale = self.patch_applier.sample_length.mean(dim=-1)

        z = self.param_atk.unsqueeze(0).repeat(imgs.shape[0], 1)
        patch = self.generative_nn.dec(z, z)

        if self.hparams.gaussian_blur:
            patch_gb = gaussian_blur(patch, [15, 15], torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)
        else:
            patch_gb = patch

        patch_val = self.color_transform(patch_gb, self.color_sampler.sample(
            imgs)) if self.color_sampler is not None else patch_gb

        if val:
            return [(patch_val, patch_loc)]

        if val_single:
            imgs_patch = self.patch_applier(imgs, patch_val)
            return [imgs_patch]

        if self.hparams.lambda_tv != 0.0:
            loss_tv = tv_loss(patch)
            self.log_loss("atk_loss_tv", loss_tv)
            self.add_loss(OptimizationStages.ATK, loss_tv * self.hparams.lambda_tv)

        if self.hparams.grad_rescale:
            patch_grad_rescale = self.grad_rescale(patch)
        else:
            patch_grad_rescale = patch

        if self.hparams.gaussian_blur:
            patch_grad_rescale = gaussian_blur(patch_grad_rescale, [15, 15],
                                               torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)

        if self.color_sampler is not None:
            patch_grad_rescale = self.color_transform(patch_grad_rescale, self.color_sampler.sample(imgs))

        imgs_patch = self.patch_applier(imgs, patch_gb)
        imgs_patch_gradscale = self.patch_applier(imgs, patch_grad_rescale)

        # self.objective_loss(imgs, imgs_patch, imgs_patch_gradscale, bboxes, "cxcywh", self.detector, mask=mask,
        #                     sample_size=sample_scale ** 2, inv_scale=task_ratio[:, 1] * tw[1],
        #                     atk_scale=task_ratio[:, 0] * tw[0], eval_mode=training_stage > 2)

        self.objective_loss(imgs, imgs_patch, imgs_patch_gradscale, bboxes, "cxcywh", self.detector, mask=None,
                            sample_size=sample_scale ** 2, inv_scale=0.0,
                            atk_scale=1.0, eval_mode=training_stage > 2)

        if visualize:
            self.visualize_cache.append(imgs.detach())
            self.visualize_cache.append(patch.detach())
            self.visualize_cache.append(imgs_patch_gradscale.detach())


def gen_perm_pair(n, device):
    a = torch.randperm(n, device=device).to(torch.int64)
    b = torch.arange(n, dtype=torch.int64, device=device)
    b[a] = b.clone()
    return a, b


class ScenarioPhysics(ScenarioDynamicPretrain):

    def __init__(self, logger, detector, generator: EnvEncoder2, generative_nn: ResVAE2, hparams, loss_cls=GeoLoss,
                 ood_detector=None, task_balacing: Optional[TaskWeightBalancing] = None):
        super().__init__(logger, detector, generator, generative_nn, hparams, loss_cls, ood_detector, task_balacing)

        self.registration_loss = RegistrationLoss(self, self.hparams)

    @property
    def num_stages(self):
        return 6

    def forward(self, imgs, bboxes, format="xywh", val=False, val_single=False, visualize=False, ratio=None,
                training_stage=0):
        if val or val_single:
            training_stage = 3
        assert imgs.shape[-1] == self.input_size
        assert bboxes.max() <= 2.0, f"Please use normalized bbox for input, cur bbox: {bboxes}"
        if (val or val_single) and ratio is None:
            ratio_list = [[1.0, 0.0], [0.5, 0.5], [0.1, 0.9]]
            return self(imgs, bboxes, format, val, val_single, ratio=ratio_list[0]) + \
                self(imgs, bboxes, format, val, val_single, ratio=ratio_list[1]) + \
                self(imgs, bboxes, format, val, val_single, ratio=ratio_list[2])

        bboxes = box_convert(bboxes, format, "cxcywh")

        self.visualize_cache = []
        patch_loc = self.get_patch_loc(bboxes)
        patch = torch.ones(imgs.shape[0], imgs.shape[1], self.patch_size, self.patch_size).to(imgs) * 0.5
        self.task_ratio_sampler.set_val_ratio(ratio)

        task_ratio = self.task_ratio_sampler(imgs)
        mask = torch.zeros_like(task_ratio[..., :1])

        mask1 = task_ratio[..., :1]

        self.patch_applier.update_affine_mat_transform(patch_loc, patch, imgs)
        img_patch, imgs_aug = self.do_aug(imgs, patch, visualize)

        self.patch_applier.reverse()
        patch_bg = self.patch_applier(patch.detach(), imgs)  # TODO: simplify this
        patch_bg[task_ratio[..., 0] > 0.9] = 0
        self.patch_applier.reverse()

        if not (val or val_single) and self.hparams.lambda_vae != 0.0:
            img_samples = imgs_aug[:, :,
                          (self.input_size // 2 - self.patch_size // 2):(self.input_size // 2 + self.patch_size // 2),
                          (self.input_size // 2 - self.patch_size // 2):(
                                  self.input_size // 2 + self.patch_size // 2)].detach()
            img_samples = img_samples[::2]

            img_samples = torch.cat([img_samples, patch_bg[::2]], dim=0)
            self.loss_generative_nn(img_samples, "gen", self.hparams.lambda_vae)

        sample_scale = self.patch_applier.sample_length.mean(dim=-1)

        # Fork
        kl_losses0, kl_losses1, patch, _ = self.nn_model(imgs_aug, None, patch_bg, task_ratio, training_stage)

        imgs_patch = self.post_process_val(imgs, patch, sample_scale)

        # task_ratio1 = self.task_ratio_sampler(imgs)
        task_ratio1 = task_ratio.clone()
        task_ratio1[task_ratio1.shape[0] // 2 : , 0] += (1 - task_ratio1[task_ratio1.shape[0] // 2 :, 0]) / 2
        task_ratio1[task_ratio1.shape[0] // 2 : , 1] = 1 - task_ratio1[task_ratio1.shape[0] // 2 : , 0]

        # perm, rev = gen_perm_pair(mask1.shape[0], mask.device)
        kl_losses0_, kl_losses1_, patch_, box_pos = self.nn_model(
            self.do_aug(imgs_patch.detach(), torch.ones_like(patch), visualize)[1], None,
            patch_bg.detach(), task_ratio1, training_stage)


        if val:
            raise NotImplementedError()

        if val_single:
            imgs_patch_ = self.post_process_val(imgs, patch_, sample_scale)
            # imgs_patch_ = self.post_process_val(imgs, patch_[rev], sample_scale)
            return [imgs_patch, imgs_patch_]

        # self.registration_loss(box_pos, patch_loc)
        # Merge
        cat = lambda x, y: torch.flatten(torch.stack([x, y], dim=1), start_dim=0, end_dim=1)
        patch, imgs, sample_scale, task_ratio, bboxes = \
            cat(patch, patch_), cat(imgs, imgs), cat(sample_scale, sample_scale), cat(task_ratio,task_ratio1), cat(
                bboxes, bboxes)
        self.patch_applier.update_affine_mat_transform(cat(patch_loc, patch_loc), patch, imgs)


        lambda_kl = getattr(self.hparams, "kl_constrain", self.hparams.lambda_vae) / 2
        if isinstance(kl_losses0, torch.Tensor):
            self.log_loss("kl_loss_env", cat(kl_losses0, kl_losses0_).mean())
        if isinstance(kl_losses1, torch.Tensor):
            self.log_loss("kl_loss_p", cat(kl_losses1, kl_losses1_).mean())
        if isinstance(kl_losses0, torch.Tensor) and not hasattr(self.hparams, "abl_kl_constrain"):
            self.add_loss(OptimizationStages.PATCH_REGULARIZATION_G,
                          cat(kl_losses0, kl_losses0_) * lambda_kl, deformable = True)
        if isinstance(kl_losses1, torch.Tensor) and not hasattr(self.hparams, "abl_kl_constrain"):
            self.add_loss(OptimizationStages.PATCH_REGULARIZATION_G,
                          cat(kl_losses1, kl_losses1_) * lambda_kl, deformable = True)

        ratio = 1 if self.task_balancing is None else self.task_balancing.ratio[0].detach()
        tw = torch.tensor([self.hparams.lambda_atk, self.hparams.lambda_distortion * ratio]).to(imgs)

        if self.hparams.lambda_tv != 0.0:
            loss_tv = tv_loss(patch)
            self.log_loss("atk_loss_tv", loss_tv)
            self.add_loss(OptimizationStages.DISTORTION, loss_tv * self.hparams.lambda_tv)

        if self.hparams.grad_rescale:
            patch_grad_rescale = self.grad_rescale(patch)
        else:
            patch_grad_rescale = patch

        if self.hparams.gaussian_blur:
            patch_grad_rescale = gaussian_blur(patch_grad_rescale, [15, 15],
                                               torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)

        if self.color_sampler is not None:
            patch_grad_rescale = self.color_transform(patch_grad_rescale, self.color_sampler.sample(imgs))

        if self.hparams.gaussian_blur:
            patch_gb = gaussian_blur(patch, [15, 15], torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)
        else:
            patch_gb = patch
        imgs_patch = self.patch_applier(imgs, patch_gb)

        imgs_patch_gradscale = self.patch_applier(imgs, patch_grad_rescale)
        self.objective_loss(imgs, imgs_patch, imgs_patch_gradscale, bboxes, "cxcywh", self.detector, mask=None,
                            sample_size=sample_scale ** 2, inv_scale=task_ratio[:, 1] * tw[1],
                            atk_scale=task_ratio[:, 0] * tw[0], eval_mode=training_stage > 2)

        if visualize:
            self.visualize_cache.append(imgs.detach())
            self.visualize_cache.append(patch.detach())
            self.visualize_cache.append(imgs_patch_gradscale.detach())

    def post_process_val(self, imgs, patch, sample_scale):
        if self.hparams.gaussian_blur:
            patch_gb = gaussian_blur(patch, [15, 15], torch.ones_like(imgs[:, 0, 0, 0]) * sample_scale / 2)
        else:
            patch_gb = patch
        patch_gb = torch.nan_to_num(patch_gb, nan=0.0, posinf=0.0, neginf=0.0)
        patch_val = self.color_transform(patch_gb, self.color_sampler.sample(
            imgs)) if self.color_sampler is not None else patch_gb
        return self.patch_applier(imgs, patch_val)

    def nn_model(self, imgs_aug, mask, patch_bg, task_ratio, training_stage):
        if training_stage == 0:
            with skip_grad(self.generator):
                patch, kl_losses0, kl_losses1, pos = self.generator(None, task_ratio=task_ratio, patch=patch_bg,
                                                               img_extra_channel=None,
                                                               detach_vae=self.hparams.detach_vae, return_pos = True,
                                                               rand_vae=self.hparams.rand_local, mask=mask)
        else:
            patch, kl_losses0, kl_losses1, pos = self.generator(imgs_aug if training_stage > 1 else None,
                                                           task_ratio=task_ratio, patch=patch_bg,
                                                           img_extra_channel=None,
                                                           detach_vae=self.hparams.detach_vae, return_pos = True,
                                                           rand_vae=self.hparams.rand_local, mask=mask)
        return kl_losses0, kl_losses1, patch, pos


if __name__ == '__main__':
    tt = torch.tensor([0, 0.2, 1, 1.1, 1.1, 0, 0]).unsqueeze(0)
    tt1 = torch.tensor([0.1, 0.2, 1, 1.1, 1.1, 0, 0]).unsqueeze(0)
    print(RegistrationLoss(None, None).gen_affine_mat(tt))
    print(RegistrationLoss(None, None).gen_affine_mat(tt1))
    print(((RegistrationLoss(None, None).gen_affine_mat(tt1) - RegistrationLoss(None, None).gen_affine_mat(
        tt)) ** 2).mean())
    # t0 = RegistrationLoss.construct_theta(torch.tensor([0,0.2,1,1.1,1.1,0,0]).unsqueeze(0))
    # t1 = RegistrationLoss.construct_theta(torch.tensor([0,0.5,1,1,0,0,0]).unsqueeze(0))
    # t0[:, :, :2] += t0[:, :, [2]]
    # t1[:, :, :2] += t1[:, :, [2]]
    # print(t0, t1)
    # print(torch.nn.MSELoss()(t0, t1))
    # print(OptimizationStages.PATCH_REGULARIZATION_G)
