import math
import warnings
from functools import partial
from typing import Optional, Union

import torch.distributed
import torchvision.datasets
from easydict import EasyDict
from pytorch_lightning.utilities import grad_norm
from torch.utils.tensorboard import SummaryWriter
# import yolov5
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.utils import ops

from dynamic_example.models.detectors import UltralyticsModel, DetectionCaller,TSEAModel
from dynamic_example.utils.detection_result_ops import ultralytice_Boxes_to_pred
from dynamic_example.models.gradient_hacker import set_loss
from dynamic_example.utils.torch_utils import freeze_bn
from dynamic_example.trainer.base_atk_model import BasePatchAttacker
from dynamic_example.models import EnvEncoder2, ResVAE2, TaskWeightBalancing, VQEncoderDecoder, ResVAETime, HookEmb
from dynamic_example.tasks.scenarios import OptimizationStages, ScenarioDynamicPretrain
from workflow.lightning_trainer import *
from det_root.utils.parser import ConfigParser
from det_root.detlib.utils import init_detector

# import mmyolo
def collate_fn(batch):
    return (torch.stack([x[0] for x in batch]),
            [([torch.tensor(i["bbox"]) for i in x[1]]) for x in batch])


def calc_padding(shape, new_shape=(640, 640)):
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    return dw, dh


class DetectionPretrain(BasePatchAttacker):
    dev_run = False  # This is a static field

    def tensorboard_logger(self, name, data):
        if self.logger is None:
            return None
        if self.is_setup:
            return
        self.log(name, data)

    def embedding_logger(self, emb, metadata, meta_names):
        self.emb_header = meta_names
        self.emb_tag_stack.extend(metadata)
        self.emb_stack = torch.cat([self.emb_stack, emb.detach()], dim = 0) if self.emb_stack.shape[0] > 0 else emb.detach()

    def pop_emb(self):
        if len(self.emb_tag_stack) == 0:
            return
        self.logger.experiment.add_embedding(self.emb_stack.float(), metadata=self.emb_tag_stack, metadata_header=self.emb_header,
                                             global_step=self.global_step)

        self.emb_stack =  torch.tensor([]).to(self.emb_stack)
        self.emb_tag_stack.clear()
    def __init__(self, conf):
        self.detector: Optional[DetectionCaller] = None
        super().__init__(conf)
        self.register_buffer("emb_stack", torch.tensor([]), persistent=False)
        self.emb_tag_stack = []
        self.emb_header = []

        self.init_detector()
        self.tmp_results = []
        self.test_clean = False
        # self.loss_atk_detector = DetLossSingle(target_clsses=[0])
        self.AAA = False
        self.BBB = False
        self.init_models()
        self.init_scenario()

        self.finetune = False

        # self._scenario.attach_controller(self.do_optimize)
        self.init_metrics(self._scenario.num_stages)
        self.warning_state = False
        self.is_setup = True
        self.scenario.setup(self.device)
        self.is_setup = False
    def setup(self, stage: str) -> None:
        if self.trainer.num_devices > 1:
            self.is_setup = True
            self.cuda()
            self.scenario.setup(self.device)
            self.is_setup = False
            for name, param in self.named_parameters():
                if isinstance(param, torch.nn.parameter.UninitializedParameter):
                    warnings.warn(f"{name} is uninitialized")
                    exit()



    def init_models(self):
        self.generator = EnvEncoder2(**self.hparams.model_config)
        if hasattr(self.hparams, "include_dec_fea"):
            self.dec_emb = HookEmb(3, 256, 16)
            self.dec_emb.attach(self.detector_raw, self.hparams.include_dec_fea)
            self.generator.attach_detector(self.detector, self.dec_emb)
        if self.hparams.use_pretrain_vq:
            self.patch_vae = VQEncoderDecoder(latent_dim=self.hparams.model_config.latent_dim, flatten = self.hparams.model_config.flatten)
        elif self.hparams.model_config.flatten and not getattr(self.hparams.model_config, "ablation_decode_t", False):
            self.patch_vae = ResVAETime(latent_dim=self.hparams.model_config.latent_dim, time_dim = self.hparams.model_config.fea_dim,
                       flatten = self.hparams.model_config.flatten, num_filters=getattr(self.hparams.model_config, "num_filters", [64, 128, 256, 512]))
        else:
            self.patch_vae = ResVAE2(latent_dim=self.hparams.model_config.latent_dim, residual_dec=getattr(self.hparams.model_config, "residual_dec", True),
                       flatten = self.hparams.model_config.flatten, num_filters=getattr(self.hparams.model_config, "num_filters", [64, 128, 256, 512]))
        if self.hparams.lambda_vae_adv > 0.0:
            self.ood_detector = ResVAE2(num_filters=[32, 64, 128, 256], latent_dim=128, flatten=True) if not self.hparams.use_gan\
                else BBoxTransformer(256)
        else:
            self.ood_detector = None
        self.generator.attach_vae(self.patch_vae)
        self.task_weight_alignment = TaskWeightBalancing()
    def init_scenario(self):
        self._scenario = ScenarioDynamicPretrain(hparams=self.hparams,
                                                 generator=self.generator,
                                                 detector=self.detector,
                                                 ood_detector=self.ood_detector,
                                                 generative_nn=self.patch_vae,
                                                 task_balacing=self.task_weight_alignment,
                                                 logger=self.tensorboard_logger)

        self._scenario.emb_logger = self.embedding_logger

    @property
    def scenario(self):
        return self._scenario

    def init_detector(self):
        if (hasattr(self.hparams, "detector") and self.hparams.detector in ["ssd", "fastrcnn", "v2", "v3", "v3-tiny", "v5"]) or (hasattr(self.hparams, "detector_cfg") and not hasattr(self.hparams, "detector")):
            if  hasattr(self.hparams, "detector_cfg"):
                cfg_path = self.hparams.detector_cfg
            else:
                cfg_path = "/home/hujin/zjk/DatkTSEA/transplant/configs/baseline/" + self.hparams.detector + ".yaml"
            cfg = ConfigParser(cfg_path)
            detector = init_detector(cfg.DETECTOR.NAME[0], cfg.DETECTOR)
            model_name = cfg.DETECTOR.NAME[0]
            print("----------------------------------------------------------------------")
            print("YOLO MODEL INIT")
            print("THE DETECTOR MODEL IS " + model_name )
            print("BEST WISHES FOR A SUCCESSFUL EXPERIMENT")
            print("----------------------------------------------------------------------")
            detector.eval()
            if isinstance(detector.detector, torch.nn.Module):
                self.add_module("detector_raw", detector.detector)
                self.force_eval_detector(detector.detector)
            shakedrop = cfg.DETECTOR.PERTURB.GATE is not None
            if shakedrop: # re-initialize original model for evaluation
                print("SHAKEDROP detected. Using Another Detector for evaluation")
                cfg1 = ConfigParser(cfg_path)
                cfg1.DETECTOR.PERTURB.GATE = None
                detector1 = init_detector(cfg1.DETECTOR.NAME[0], cfg1.DETECTOR)
                self.add_module("detector_raw1", detector1.detector)
                self.force_eval_detector(detector1.detector)
                self.detector = TSEAModel(detector, detector1, self.hparams)
            else:
                self.detector = TSEAModel(detector, None, self.hparams)
        else:
            model_name = self.hparams.detector
            if model_name.startswith("rtdetr"):
                from ultralytics import RTDETR
                detector = RTDETR(os.path.join(self.hparams.path_to_ckpts, model_name + '.pt')).model
                # self.add_module("detector_raw", detector)
            else:
                detector = YOLO(os.path.join(self.hparams.path_to_ckpts, model_name + '.pt')).model
            
            print("----------------------------------------------------------------------")
            print("YOLO MODEL INIT")
            print("THE DETECTOR MODEL IS " + model_name)
            print("BEST WISHES FOR A SUCCESSFUL EXPERIMENT")
            print("----------------------------------------------------------------------")
            self.force_eval_detector(detector)
            self.add_module("detector_raw", detector)
            self.detector = UltralyticsModel(detector, self.hparams)

    def force_eval_detector(self, detector):
        detector.eval()
        detector.train = lambda x: detector
        for param in detector.parameters():
            param.requires_grad = False

        def void(*args, **kwargs):
            return {}

        detector.state_dict = void
        detector = detector.to(self.device)

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        if not isinstance(loss, torch.Tensor):
            return
        if not DetectionPretrain.dev_run and torch.isnan(loss):
            warnings.warn("NAN loss occurred! step jumped.")
            return
        super().manual_backward(loss, *args, **kwargs)

    def do_optimize(self, img):
        optimG, optimVAE,  optimD, optim_task = self.optimizers()

        loss_atk = self.scenario.fetch_total_loss(OptimizationStages.ATK)
        gradscale_atk = self.scenario.fetch_total_loss(OptimizationStages.ATK_GRADSCALE)
        loss_distortion = self.scenario.fetch_total_loss(OptimizationStages.DISTORTION)  # * self.hparams.lambda_distortion
        stat_distortion = self.scenario.fetch_total_loss(OptimizationStages.DISTORTION_STAT_UNNORMED)  # * self.hparams.lambda_distortion
        stat_atk = self.scenario.fetch_total_loss(OptimizationStages.ATK_STAT_UNNORMED)  # * self.hparams.lambda_distortion

        loss_reg_G = self.scenario.fetch_total_loss(OptimizationStages.PATCH_REGULARIZATION_G)
        # if isinstance(loss_reg_G, torch.Tensor):
        #     loss_reg_G = loss_reg_G.sum()
        loss_D = self.scenario.fetch_total_loss(OptimizationStages.PATCH_REGULARIZATION_D)

        loss_vae = self.scenario.fetch_total_loss(OptimizationStages.VAE)
        self.task_weight_alignment.forward(stat_atk, 0)
        self.task_weight_alignment.forward(stat_distortion, 1)

        tot_grad_norm = 0
        # Train High Level Mapping
        if self.shall_train_env():

            # Train GAN D
            optimG.zero_grad()
            self.apply(partial(set_loss, loss_scale=gradscale_atk, stage=0))
            loss = 0
            if not isinstance(loss_reg_G, torch.Tensor):
                loss_reg_G = torch.tensor(loss_reg_G).to(self.device)
            loss = loss + loss_reg_G.sum()
            if self.shall_train_atk():
                loss = loss + loss_atk.sum()
                # self.manual_backward(loss,
                #                      retain_graph=self.hparams.lambda_vae > 0.0 or self.hparams.lambda_distortion > 0.0)
                # loss = 0

            # tot_grad_norm = self.stablelize(optimG, tot_grad_norm, self, "/optim_G_ATK/", norm_type=2)

            self.apply(partial(set_loss, loss_scale=None, stage=1))

            if self.hparams.lambda_distortion > 0.0:
                # loss = loss_distortion.mean()# * self.task_weight_alignment.ratio[0].detach()
                loss = loss + loss_distortion.sum()
            if self.hparams.lambda_vae > 0.0:
                loss = loss + loss_vae.sum()
            self.manual_backward(loss)


            # if self.hparams.lambda_vae > 0.0:
            #     self.apply(partial(set_loss, loss_scale=loss_atk, stage=2))
            #     loss = loss_vae.mean()
            #     self.manual_backward(loss)

            tot_grad_norm = self.stablelize(optimG, tot_grad_norm, self, "/optim_G/")
            optimG.step()

            # Train GAN D

            if self.hparams.lambda_vae_adv > 0 and isinstance(loss_D, torch.Tensor) and self.shall_train_atk():
                optimD.zero_grad()
                self.manual_backward(loss_D.mean())


                tot_grad_norm = self.stablelize(optimD, tot_grad_norm, self.ood_detector, "/optim_D/")
                optimD.step()



        elif self.hparams.lambda_vae > 0.0:
            optimVAE.zero_grad()
            self.manual_backward(loss_vae.sum())
            tot_grad_norm = self.stablelize(optimVAE, tot_grad_norm, self.patch_vae, "/optim_VAE/")
            optimVAE.step()
        else:
            assert self.shall_train_env(), "Wrong Config: start_train_env should be 0 when lambda_vae == 0.0."
            # if self.hparams.lambda_vae > 0.0:
            #     loss +=

        self.log("loss_atk", loss_atk.mean(), prog_bar=True)
        self.log("loss_atk_high_ratio", loss_atk[self.scenario.task_ratio_sampler.get_atk_batches(img)].mean(),
                 prog_bar=True)
        self.log("loss_distortion", loss_distortion.mean(), prog_bar=True)
        self.log("task_skewness", self.task_weight_alignment.get_skewness(1) - self.task_weight_alignment.get_skewness(0))
        self.log("task_ratio", self.task_weight_alignment.ratio[0])

        if isinstance(loss_vae, torch.Tensor):
            self.log("loss_vae", loss_vae.mean(), prog_bar=True)


        # Train Hyperparam Finetuner
        if self.hparams.grad_task_norm and self.shall_train_atk() and not self.finetune:
            optim_task.zero_grad()

            self.manual_backward(self.task_weight_alignment.loss())
            optim_task.step()

        if self.hparams.log_norms:
            self.log("grad_inf_norm_total", tot_grad_norm)


    def stablelize(self, optim: torch.optim.Optimizer, tot_grad_norm, module: torch.nn.Module, param_name, norm_type: Union[str | int | float] = "inf", do_clip = True):
        if self.hparams.log_norms:
            norms = grad_norm(module, norm_type=norm_type, group_separator=param_name)
            if norms:
                if norm_type == "inf":
                    tot_grad_norm = max(float(norms.pop("grad_inf_norm_total")), tot_grad_norm)
                self.log_dict(norms)
                
        # benchmark: terminated
        # for param_group in optim.param_groups:
        #     for param in param_group['params']:
        #         param_gradient = param.grad
        #         if param_gradient is not None:
        #             if torch.isinf(param_gradient).any() or torch.isnan(param_gradient).any():
        #                 warnings.warn("nan grad detected! clip to zero.")
        #                 print("nan grad detected! clip to zero", torch.isinf(param_gradient).sum(), torch.isnan(param_gradient).sum())

        #                 param.grad = torch.nan_to_num(param_gradient, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(self.hparams, "gradient_clip_val") and do_clip:
            self.clip_gradients(optim, self.hparams.gradient_clip_val, gradient_clip_algorithm="value")
        return tot_grad_norm

    def training_step(self, batch, i_batch):

        img, bboxes = batch

        vis = int(math.ceil(self.trainer.max_epochs / 256))
        vis = vis if vis > 0 else 1
        try:
            training_stage = 0
            if self.shall_train_env(): training_stage += 1
            if self.shall_train_atk(): training_stage += 1
            if self.finetune and self.shall_train_atk_fintune(): training_stage = 3
            ratio = self.hparams.val_ratio if hasattr(self.hparams, "val_ratio") and self.finetune else None
            ratio = self.hparams.fixed_ratio if hasattr(self.hparams, "fixed_ratio") else ratio
            self.scenario(img, bboxes,
                          visualize=DetectionPretrain.dev_run or (i_batch == 0 and self.current_epoch % vis == 0) or not self.BBB,
                          training_stage=training_stage,
                          ratio = ratio)
            all_loss = self.scenario.fetch_total_loss(all=True)
            if torch.isnan(all_loss).any() or torch.isinf(all_loss).any():
                if self.warning_state:
                    raise RuntimeError("Forward Failed / NAN for 2 times, Terminate!")
                warnings.warn("NAN occurred")
                self.warning_state = True
                return None
            self.warning_state = False
        except Exception as e:
            self.scenario.reset_losses()
            if DetectionPretrain.dev_run:
                raise RuntimeError("Training step exception") from e
            else:
                if self.warning_state:
                    # restore last checkpoint
                    raise RuntimeError("Forward Failed / NAN  for 2 times, Terminate!") from e
                    # self.load_from_checkpoint(self.trainer.ckpt_path)
                print("Forward Failed. skipped")
                traceback.print_tb(e.__traceback__)
                warnings.warn(str(e))
                self.warning_state = True
                return


        self.do_optimize(img)

        self.scenario.reset_losses()
        vis = self.scenario.visualize_cache
        # if self.global_rank != 0:
        #     return

        for i, img in enumerate(vis):
            import workflow.lightning_trainer
            workflow.lightning_trainer.plot(img, f"train{i} rnk{self.global_rank}.jpg")
            img_grid = torchvision.utils.make_grid(img.float(), nrow=8, normalize=False, scale_each=False)
            sp = img_grid.shape[-2] / img_grid.shape[-1]
            img_grid = torchvision.transforms.Resize((int(1024 * sp), 1024), antialias=True)(img_grid)
            self.logger.experiment.add_image(f"input img{i}_rnk{self.global_rank}", img_grid, self.global_step)
        self.BBB = True
        self.scenario.visualize_cache = None
        # exit()

    def shall_train_atk(self):
        return self.global_step  >= self.hparams.start_train_atk # BEFORE * NUM OPTIM STEPSS

    def shall_train_env(self):
        return self.global_step  >= self.hparams.start_train_env # BEFORE * NUM OPTIM STEPSS

    def shall_train_atk_fintune(self):
        return hasattr(self.hparams, "start_train_atk_finetune_epoch") and self.current_epoch >= self.hparams.start_train_atk_finetune_epoch
    def attack_clean(self, imgs, bboxes, inplace=False, conf_thresh=0.5, cls=[1], remap_bbox=None, multi_stage=True):

        # patch bboxes to batch size
        # stack & result initialization
        cur_bs = 0
        batch_imgs = []
        batch_boxes = []
        batch_ids = []
        cls_mapping = remap_bbox
        cls = set(cls)

        B, C, H, W = imgs.shape
        num_stages = self.scenario.num_stages
        patches_results = []
        for s in range(num_stages):
            patches_results.append([])
            for i in range(B):
                patches_results[s].append([])

        # import time
        # t = time.time()
        # for each item in batch
        for i in range(B):
            boxes: dict = bboxes[i]
            ls = list((boxes["scores"] >= 0.5).cpu())
            box_data = [(b, cls) for b, cls, flag in zip(boxes["boxes"], boxes["labels"], ls) if flag]
            # boxes.data = boxes.data.clone().contiguous()

            # torch.clip_(cocobbox[:, 2:], max=min(W, H) / 2)
            # cocobbox = ops.xyxy2ltwh(boxes.xyxy)

            for j in range(len(box_data)):
                # stack push
                if cls.__contains__(int(box_data[j][1]) if cls_mapping is None else cls_mapping[int(box_data[j][1])]):
                    batch_ids.append(i)
                    cocobbox = box_data[j][0].clone() # xywh
                    # cocobbox[:, 2:] = torch.clip(cocobbox[:, 2:], max=min(W, H) / 2)
                    # cocobbox = ops.xyxy2ltwh(cocobbox)[0]
                    batch_boxes.append(cocobbox)
                    # batch_boxes.append(torch.tensor([1280 / 2 - 200, 704 / 2 - 100,400, 200]).to(cocobbox.device))
                    cur_bs += 1

                # stack pop & parallel process & store to rendering queue
                if cur_bs == 8 or (i == imgs.shape[0] - 1 and j == len(box_data) - 1 and cur_bs > 0):
                    input_img = imgs[batch_ids]
                    input_bbox = torch.stack(batch_boxes, dim=0)

                    batch_boxes = []

                    patches = self.scenario(torch.clone(input_img), input_bbox, val=True)
                    for s in range(num_stages):
                        p, box = patches[s]
                        # print(p.shape, i)
                        for k in range(len(batch_ids)):
                            patches_results[s][batch_ids[k]].append((p[k], box[k]))

                    batch_ids = []
                    cur_bs = 0
        # print(time.time() - t)
        # t = time.time()
        # render results
        results = []
        for x in range(num_stages):
            results.append(imgs.clone())
        for s in range(num_stages):
            imgs = results[s]
            for i in range(B):
                # print(B)
                # print(len(patches_results[s][i]))
                for patch, ccwhrrr in patches_results[s][i]:
                    p = patch.unsqueeze(0)
                    self.applier.update_affine_mat_transform(ccwhrrr.unsqueeze(0), p, imgs[[i]])
                    imgs[[i]] = self.applier(imgs[[i]], p)
                    # pil: PIL.Image.Image = torchvision.transforms.ToPILImage()(patch[0].detach())
                    # pil.save(getCfg().get_log_dir() + "/patch.png")
        # print(time.time() - t)
        if not multi_stage:
            return results[-1]
        return results

    def validation_step(self, batch, i_batch):
        # if i_batch > 2:
        #     return

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.set_grad_enabled(False):

            imgs, bboxes, targets = batch
            w, h = imgs.shape[-1], imgs.shape[-2]
            self.eval()
            iterations = self.hparams.validation_sample_cnt if hasattr(self.hparams, "validation_sample_cnt") else 1
            epoch = 9999999999999 if hasattr(self, "loaded_epoch") else self.current_epoch
            epoch_lim = self.trainer.max_epochs if self.trainer.max_epochs is not None else -1
            if self.hparams.clean_val or (epoch < epoch_lim - 1):
                iterations = 1
            for it_num in range(iterations):
                if self.hparams.clean_val:
                    results = [imgs.clone() for _ in range(self.scenario.num_stages)]
                elif self.hparams.single_val:
                    with torch.no_grad():
                        with torch.inference_mode():
                            starter.record()
                            results = self.scenario(imgs, bboxes, val_single=True, format="xywh")
                            ender.record()
                            torch.cuda.synchronize()
                            curr_time = starter.elapsed_time(ender)
                            if i_batch >= 5:
                                self.add_time(curr_time)

                else:
                    bboxes = [Boxes(torch.cat([ops.ltwh2xyxy(x), torch.ones_like(x[..., :2])], dim=-1), (h, w)) for x in
                              bboxes]
                    with torch.no_grad():
                        results = self.attack_clean(imgs, bboxes)
                if not self.AAA:
                    for i in range(self.scenario.num_stages):
                        import workflow.lightning_trainer
                        imgs1 = results[i].detach()
                        preds = self.generate_bboxes(imgs1)
                        workflow.lightning_trainer.plot(imgs1, f"atk{i} rnk{self.global_rank}.jpg", preds=preds,
                                                        box_scale=self.hparams.input_pixel)

                    # from ultralytics.utils.plotting import plot_images
                    #
                    # boxes: Boxes = bboxes[0]
                    # cls = boxes.cls
                    # idx = torch.zeros_like(cls)

                    # plot_images(imgs[0].unsqueeze(0), idx,  cls, boxes.ccwh,fname = getCfg().get_log_dir() +"/bbox_orig.png" )
                    self.AAA = True

                for i, imgs_atked in enumerate(results):
                    preds = self.generate_bboxes(imgs_atked)
                    self.update_metrics(imgs, imgs_atked, preds, targets, i, fuse = it_num == iterations - 1)

    def on_validation_end(self):
        super().on_validation_end()
        self.pop_emb()
        self.AAA = False

    def on_test_end(self) -> None:
        super().on_test_end()
        self.pop_emb()

    def test_step(self, batch, i_batch):
        return self.validation_step(batch, i_batch)

    def generate_bboxes(self, imgs):
        return self.detector.pred(imgs)

    def predict(self, imgs, embds=None):
        return self.generator(imgs, embds)

    def configure_optimizers(self):
        optimG = torch.optim.Adam(list(self.generator.parameters()) + list(self.patch_vae.parameters()),
                                  lr=self.hparams.lr)
        optimVAE = torch.optim.Adam(list(self.patch_vae.parameters()),
                                  lr=self.hparams.lr)
        if hasattr(self, "ood_detector") and self.ood_detector is not None:
            optim_D = torch.optim.Adam(list(self.ood_detector.parameters()), lr=self.hparams.lr / 2)
        else:
            optim_D = optimVAE
        optim_task = torch.optim.Adam(list(self.task_weight_alignment.parameters()), lr=self.hparams.lr)
        return optimG, optimVAE, optim_D, optim_task


if __name__ == '__main__':

    init("local_yolo", "test01")
    model = DetectionPretrain(getCfg().config.train_model).cuda().eval()
    trainer = Trainer()
    # train(model, None)
    # trainer.(model, dataloaders = model.test_dataloader())
    data = model.train_dataloader()._get_iterator().__next__()
    print(data)
    input = data[0][:2].cuda(), data[1][:2].cuda()

    # DetectionPretrain.dev_run = True
    model.train()
    model.training_step(input, 0)
    from torchvision.utils import save_image

    for i, x in enumerate(model.scenario.visualize_cache):
        save_image(x, os.path.join(getCfg().get_log_dir(), f"plot_stage{i}.png"))
