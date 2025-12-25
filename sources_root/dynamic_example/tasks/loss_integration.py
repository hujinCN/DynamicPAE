import torch
import torch.nn
import torchvision.transforms

from dynamic_example.models import ResVAE2, skip_grad
from dynamic_example.models.detectors import DetectionCaller
from dynamic_example.models.losses import LPIPS, DATKLoss, \
    DetectionLoss, reduction
from dynamic_example.tasks.scenarios import OptimizationStages, SenarioLosses


class GANLossVanilla(SenarioLosses):
    def __init__(self, parent_scenario, hparams, discrimmator):
        super().__init__(parent_scenario)

        # self._distortion = MS_SSIM(win_size=5)
        # self.distortion = (lambda x, y: 1 - self._distortion(x, y))

        self.discrimmator = discrimmator
        self.hparams = hparams


    def forward(self, real, fake_D, fake_G = None,  ratio = 1.0):
        if not self.training:
            return
        if fake_G is None:
            fake_G = fake_D
        with skip_grad(self.discrimmator):
            loglikehood = reduction(-torch.log(torch.sigmoid(self.discrimmator(fake_G))))
            self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, reduction(loglikehood).sum() * ratio, deformable=True)

        real, fake_D = torch.sigmoid(self.discrimmator(real.detach())), 1 - torch.sigmoid(self.discrimmator(fake_D.detach()))

        real_ls = -reduction(torch.log(real)).sum()
        loss = real_ls
        fake_ls = -reduction(torch.log(fake_D)).sum()
        loss += fake_ls
        # self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, loglikehood.mean())
        self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_D, loss)
        self.parent_scenario.log_loss("GAN correct logits", loglikehood.mean())
        self.parent_scenario.log_loss("GAN ALL", loss.mean())




class GANLoss(SenarioLosses):
    def __init__(self, parent_scenario, hparams, discrimmator):
        super().__init__(parent_scenario)

        # self._distortion = MS_SSIM(win_size=5)
        # self.distortion = (lambda x, y: 1 - self._distortion(x, y))

        self.discrimmator = discrimmator
        self.hparams = hparams
        self.lambda_gp = self.hparams.lambda_gp if hasattr(self.hparams, "lambda_gp") else 1.0


    def cal_gp(self, fake_imgs, real_imgs):
        r = fake_imgs.shape
        for i in range(1, len(r)):
            r[i] = 1
        r = torch.rand(size=r, device=real_imgs.device)
        x = (r * real_imgs + (1 - r) * fake_imgs)
        print(fake_imgs.shape, real_imgs.shape, r.shape, x.shape)
        x = torch.autograd.Variable(x, requires_grad=True)
        d = reduction(self.discrimmator(x))
        g = torch.autograd.grad(
            outputs=d,
            inputs=x,
            grad_outputs=torch.ones_like(d),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def forward(self, real, fake_D, fake_G,  ratio = 1.0):
        if not self.training:
            return
        with skip_grad(self.discrimmator):
            loglikehood = reduction(self.discrimmator(fake_G))
            self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, reduction(loglikehood).sum() * ratio, deformable=True)

        real, fake_D = self.discrimmator(real.detach()), self.discrimmator(fake_D.detach())
        real_ls = reduction(real).sum()
        loss = real_ls
        fake_ls = reduction(fake_D).sum()
        loss -= fake_ls
        # self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, loglikehood.mean())
        self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_D, loss)
        self.parent_scenario.log_loss("GAN correct logits", loglikehood.mean() / (real_ls / real.shape[0] + fake_ls / fake_D.shape[0]) )
        self.parent_scenario.log_loss("GAN ALL", real_ls / real.shape[0] - fake_ls / fake_D.shape[0])
        # d = - self.discrimmator(real.detach()).mean() + self.discrimmator(fake.detach()).mean()
        # self.parent_scenario.log_loss("WGAN_D", d)
        # self.parent_scenario.add_loss(OptimizationStages.GAN_D, d / 100)
        #
        gp = self.cal_gp(fake_D.detach(), real.detach()[:fake_D.shape[0]]) * self.lambda_gp
        # #
        self.parent_scenario.log_loss("WGAN_GP", gp)
        self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_D, gp)
        #
        # d = - self.discrimmator(fake).mean()
        # self.parent_scenario.log_loss("WGAN_G", d)
        # self.parent_scenario.add_loss(OptimizationStages.GAN_G, d / 100)


class TrainVAE(SenarioLosses):
    def __init__(self, parent_scenario, vae: ResVAE2, ood_vae: ResVAE2):
        super().__init__(parent_scenario)
        self.vae = vae
        self.ood_vae = ood_vae
        self.distortion = torch.nn.MSELoss(reduction='none')
        self.distortion2 = LPIPS(net_type="alex")

    def forward(self, patches, name, ratio):
        pp = patches.detach()
        recon, kl_loss = self.vae(pp)
        # loss_tv = tv_loss(recon)
        dis = reduction(self.distortion(recon, pp)) + self.distortion2(recon, pp)
        self.parent_scenario.log_loss(f"vae_kl_{name}", kl_loss)
        self.parent_scenario.log_loss(f"vae_d_{name}", dis)
        # self.parent_scenario.log_loss(f"vae_tv_{name}", loss_tv)
        self.parent_scenario.add_loss(OptimizationStages.VAE, ratio * (kl_loss + dis ))  # todo: move 0.1 to config
        return recon

    def forward_adv(self, patches_anchor, patches_exploit, ratio, name="adv"):
        if self.ood_vae is not None:
            with skip_grad(self.ood_vae):
                recon, kl_loss = self.ood_vae(patches_exploit)
            recon_2, kl_loss_2 = self.ood_vae(patches_anchor.detach())
            dis = self.distortion2(recon, patches_exploit)
            dis_2 = self.distortion2(recon_2, patches_anchor.detach())
            self.parent_scenario.log_loss(f"vae_kl_{name}", kl_loss)
            self.parent_scenario.log_loss(f"vae_d_{name}", dis)
            self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_G, ratio * (- kl_loss -  dis), deformable=True)
            self.parent_scenario.add_loss(OptimizationStages.PATCH_REGULARIZATION_D, kl_loss_2 + dis_2)

        # torchvision.utils.save_image(pp, "input.png")
        # torchvision.utils.save_image(recon, "recon.png")
        # exit()



class ObjectiveLossesWithGrad(SenarioLosses):
    def __init__(self, parent_scenario, hparams, loss_cls = DATKLoss, discrim = None):
        super().__init__(parent_scenario)
        # self.distortion = torch.nn.MSELoss()

        # self._distortion = MS_SSIM()
        # self.distortion = (lambda x, y: 1 - self._distortion(x, y))
        # self.distortion = torch.nn.MSELoss()
        self.distortion = LPIPS()
        self.distortion2 = torch.nn.MSELoss(reduction='none')

        self.hparams = hparams
        self.det_loss: DetectionLoss = loss_cls(hparams.det_loss)
        try:
            self.gan_loss = GANLossVanilla(parent_scenario, hparams, discrim) if self.hparams.lambda_vae_adv != 0 and self.hparams.use_gan else None
        except:
            pass

        self.transform = torchvision.transforms.Resize(self.hparams.detect_pixel, antialias=True)

    def forward(self, img, img_patch, img_patch_gradscale, gt_bbox, gt_bbox_format, model: DetectionCaller, idx_target = None, mask = None,
                sample_size = None, inv_scale = None, atk_scale = None, eval_mode = False):
        if not self.training:
            return

        pred_patch = model(img_patch_gradscale, eval_mode = eval_mode)  # ccwh[prob(cls1), prob(cls2)....]

        # yolo_loss = self.bbox_sup_loss(label_pred.float(), ccwh_patch.float())
        distortion = self.distortion(img.float(), img_patch.float())
        distortion2 = reduction(self.distortion2(img.float(), img_patch.float()))

        if (isinstance(atk_scale, torch.Tensor) and (atk_scale == 0.0).all()) or (atk_scale is 0):
            det_loss = torch.zeros_like(distortion), torch.zeros_like(distortion)
        else:
            det_loss = self.det_loss(pred_patch.float(), gt_bbox.float(), pred_format="cxcywh", label_format=gt_bbox_format,
                                     logger=self.parent_scenario.logger, idx_target = idx_target, mask = mask)

        # not yolo --nan
        loss_distortion = (distortion + distortion2)  # * self.hparams.lambda_distortion
        # loss_distortion = (distortion2)  # * self.hparams.lambda_distortion

        loss_attack, loss_atk_gradscale = det_loss
        # if sample_size is not None:
        #     loss_distortion = loss_distortion * sample_size
        #     if not self.hparams.grad_rescale:
        #         loss_attack = loss_attack * sample_size

        #
        # if isinstance(loss_atk_gradscale, torch.Tensor):
        self.parent_scenario.add_loss(OptimizationStages.ATK_STAT_UNNORMED, torch.zeros_like(loss_distortion).detach() + 1.0)
        self.parent_scenario.add_loss(OptimizationStages.DISTORTION_STAT_UNNORMED, loss_distortion.detach())

        if inv_scale is not None:
            loss_distortion = loss_distortion * inv_scale

        if atk_scale is not None:
            loss_attack = loss_attack * atk_scale
            loss_atk_gradscale = loss_atk_gradscale * atk_scale

        self.parent_scenario.log_loss("distortion", distortion)
        self.parent_scenario.log_loss("distortion2", distortion2)
        self.parent_scenario.log_loss("detection", loss_attack)


        # self.parent_scenario.log_loss("yolo", yolo_loss)

        self.parent_scenario.add_loss(OptimizationStages.DISTORTION, loss_distortion)
        self.parent_scenario.add_loss(OptimizationStages.ATK, loss_attack)
        self.parent_scenario.add_loss(OptimizationStages.ATK_GRADSCALE, loss_atk_gradscale)


    def pred(self, model, img_input):
        img_input = self.transform(img_input)
        pred = model(img_input)
        if isinstance(pred, tuple):
            pred = pred[0]
            pred[:, :4] /= self.hparams.detect_pixel
        return pred


class LossTorchATK(SenarioLosses):
    def __init__(self, parent_scenario, hparams, loss_cls):
        super().__init__(parent_scenario)
        # self.distortion = torch.nn.MSELoss()

        self.distortion2 = torch.nn.MSELoss()
        self.hparams = hparams
        self.det_loss: DetectionLoss = loss_cls(hparams.det_loss)
        # self.det_loss = DetLossSurpressCRBBOX(target_clsses=[0])

        self.transform = torchvision.transforms.Resize(self.hparams.detect_pixel, antialias=True)

    def forward(self,
                img_patch_cam_det,  # detection camera distortion
                bbox_gt, detection_model, gt_format="xywh"):

        pred_patch = detection_model(img_patch_cam_det)  # ccwh[prob(cls1), prob(cls2)....]
        # pred_orig = self.pred(detection_model, img_orig_cam_det.detach()).permute(0, 2, 1)
        det_loss = self.det_loss(pred_patch.float(), bbox_gt.float(), pred_format="cxcywh", label_format=gt_format,
                                 logger=self.parent_scenario.logger)[0]
        # yolo_loss = self.bbox_sup_loss(label_pred.float(), ccwh_patch.float())  # * 0.9
        # not yolo --nan
        loss_attack = (det_loss)
        return loss_attack
