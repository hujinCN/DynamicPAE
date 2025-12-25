import math
from typing import Literal, Tuple, Union

import torch
import torch.nn as nn
import torchmetrics
import torchvision.models
import torch.nn.functional as F
from einops import rearrange, einsum
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update
from torchvision.ops import box_convert


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original_logits, attacked_logits):
        p1 = F.softmax(original_logits.float(), dim = -1)
        log_p2 = F.log_softmax(attacked_logits.float(), dim = -1)
        return torch.sum(-p1 * log_p2, dim=-1).mean()

class CrossEntropyUniform(nn.Module):
    def __init__(self, relax_ratio = 0.001):
        super().__init__()
        self.p_randomness = relax_ratio

    def forward(self, original_logits, attacked_logits, reduction = True):
        p1 = F.softmax(original_logits.float(), dim = -1)
        p2 = F.softmax(attacked_logits.float(), dim=-1)
        p2 = p2 + torch.ones_like(p2) * 0.5 * (self.p_randomness / p2.shape[-1])
        p2 /= p2.sum(dim = -1, keepdim = True)
        log_p1 = torch.log(p1)
        log_p2 = torch.log(p2)
        if reduction:
            return torch.sum(-p1 * log_p2, dim=-1).mean() - torch.sum(-p1 * log_p1, dim=-1).mean()
        else:
            return  torch.sum(-p1 * log_p2, dim=-1) - torch.sum(-p1 * log_p1, dim=-1)

# class Proj(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, sample_logits, source_logits):
#         p1 = F.softmax(sample_logits, dim = -1)
#         log_p2 = F.log_softmax(source_logits, dim = -1)
#         return torch.sum(-p1 * log_p2, dim=-1)
# class ObjectDetection:
#     def __init__(self):
#         pass
#
#     def attention_loss(self):


class GaussianREG(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.GaussianNLLLoss(reduction="none")

    def forward(self, prob):
        return reduction(self.loss(prob, torch.zeros_like(prob), torch.ones_like(prob)))


class GradCam():
    hook_a, hook_g = None, None

    hook_handles = []

    def hack_forward(self, *args, **kwargs):
        output = self.original_forward(args, kwargs)
        self.hook_a = output
        return output.detach()


    def __init__(self, model, conv_layer, use_cuda=True, original_loss = None, norm = None):

        # self.model = model
        # self.use_cuda = use_cuda
        self.original_forward = model._modules.get(conv_layer).forward
        self.norm = norm
        model._modules.get(conv_layer).forward = self.hack_forward
        self.hook_handles.append(model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        self.hook_handles.append(model._modules.get(conv_layer).register_forward_hook(self._hook_a))

        self._relu = True
        self._score_uesd = True
        self.hook_handles.append(model._modules.get(conv_layer).register_backward_hook(self._hook_g))

        self.original_loss = original_loss

    def _hook_a(self, module, input, output):
        self.hook_a = output

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def _hook_g(self, module, grad_in, grad_out):
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]


    def _get_weights(self, class_idx, scores):

        self._backprop(scores, class_idx)

        return self.hook_g.squeeze(0).mean(axis=(1, 2))

    def __call__(self, *args, **kwargs):
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        loss_ori = self.original_loss(*args, **kwargs)
        loss_ori.backward(retain_graph=True)

        # grad = self.hook_g.mean(axis=1) # B H W

        cam = self.hook_g * self.hook_a

        # pred = F.softmax(scores)[0, class_idx]
        # weights = self._get_weights(class_idx, scores)
        # cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        # cam_np = cam.data.cpu().numpy()
        # cam_np = np.maximum(cam_np, 0)
        # cam_np = cv2.resize(cam_np, input.shape[2:])
        # cam_np = cam_np - np.min(cam_np)
        # cam_np = cam_np / np.max(cam_np)
        return cam

    def clean_cache(self):
        self.hook_a, self.hook_g = None, None







#### Following uses this reduction


def reduction(loss, type = None):
    """
    Default: Left the bs dim not reduced for custom post-processing. Mean over other dims.
    @param loss:  Loss Mat
    @param type: None, "mean","sum", ... see below
    @return: reduced loss
    """
    shape = len(loss.shape)
    if type == None:
        if shape <= 1:
            return loss
        return loss.mean(dim = list(range(1, shape)))
    if type == "mean":
        return loss.mean()
    if type == "sum":
        return loss.mean() * loss.shape[0]
    if type == "sum_each":
        return loss.sum()
    if type == "sum_batch":
        if shape <= 1:
            return loss
        return loss.sum(dim = list(range(1, shape)))

from torchvision.ops._utils import _loss_inter_union
# class DetLossSingle(nn.Module):
#     def __init__(self, sigma = 1.0, target_clsses=None, conf_thresh = 0.1, iou_thresh = 0.1):
#         self.sigma = sigma
#         self.target_clsses = target_clsses
#         if self.target_clsses is not None:
#             self.target_clsses = [i + 4 for i in self.target_clsses]
#         self.conf_thresh = conf_thresh
#         self.iou_thresh = iou_thresh
#         super().__init__()
#
#     def forward(self, pred, label):
#         # pred: (bs, num_box, num_cls + 4) xyxy
#         # label: (bs, 4) xyxy
#         pred = pred.float()
#         label = label.float()
#         label = label.unsqueeze(1).repeat(1, pred.shape[1], 1)
#
#         # print(pred[0, 0])
#         # print(pred[0,: , 1+4].max())
#         # print(pred[0,: , 1+4].min())
#         # print(pred[0,: , 1+4].mean())
#         if self.target_clsses is None:
#             conf = torch.sum(pred[:, :, 4:], dim = -1).clip_(min = 0.1)
#         else:
#             conf = torch.sum(pred[:, :, self.target_clsses], dim = -1) #.clip_(min = 0.1)
#         inter, union = _loss_inter_union(label, pred[:, :, :4])
#         mask = union > 0
#         inter[mask] /= union[mask]
#         # noinspection PyTypeChecker
#         mask = torch.logical_and(inter > self.iou_thresh, conf > self.conf_thresh)
#         # return torch.log(torch.clip_(torch.sum(conf * inter * mask, dim = -1), min = 1e-7))
#
#
#         return reduction((torch.log(1 - conf) * inter * mask).sum(dim = -1))
#

def tv_loss(image):
    # Calculate the horizontal differences
    horizontal_diff = (image[:, :, :, :-1] - image[:, :, :, 1:]) **2

    # Calculate the vertical differences
    vertical_diff = (image[:, :, :-1, :] - image[:, :, 1:, :]) ** 2

    # Calculate the sum of differences
    total_variation = reduction(horizontal_diff) + reduction(vertical_diff)

    return total_variation

# class CosSimLosses(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass
#
#     def forward(self, fea1, fea2, dim = 1):
#         fea1, fea2 = fea1.float(), fea2.float()
#         ret = (fea1 * fea2).sum(dim = dim) / (torch.norm(fea1, p=2, dim = dim) * torch.norm(fea2, p=2, dim = dim)).clip(min = 1e-7)
#         return ret.mean()

def loss_nap(iou, conf, conf_cls):
    return conf.max(dim = 1)


def loss_logit_probs(iou, conf, conf_cls, hparams):
    return conf.max(dim = 1)





def learned_perceptual_image_patch_similarity(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex"
) -> torch.Tensor:
    net = _NoTrainLpips(net=net_type)
    loss, total = _lpips_update(img1, img2, net, True)
    return loss
class LPIPS(nn.Module):
    net_global = {}
    def __init__(self, net_type: Literal["alex", "vgg", "squeeze"] = "alex"):
        super().__init__()
        if hasattr(LPIPS.net_global, net_type):
            self.net = LPIPS.net_global[net_type]
        else:
            self.net = _NoTrainLpips(net=net_type).eval()
            for p in self.net.parameters():
                p: torch.nn.Parameter
                p.requires_grad = False
            LPIPS.net_global[net_type] = self.net

    def forward(self,
                img1: torch.Tensor,
                img2: torch.Tensor):
        loss, total = _lpips_update(img1, img2, self.net, True)
        return reduction(loss)


class DetectionLoss(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.supress_clsses = hparams.supress_clsses
        if self.supress_clsses is not None:
            if isinstance(self.supress_clsses, int):
                self.supress_clsses = [self.supress_clsses]
            # self.supress_clsses = [i + 4 for i in self.supress_clsses]
        self.tgt_classes = None

    def forward(self, pred, label, pred_format = "cxcywh", label_format = "xyxy", mask = None, logger = None, **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor],torch.Tensor]:
        label = label.float()
        pred = pred.float()

        pred_pos = box_convert(pred[..., :4], pred_format, "xyxy")
        label = box_convert(label[..., :4], label_format, "xyxy")

        # pred_pos = box_convert(pred[..., :4].detach(), pred_format, "cxcywh")
        # label = box_convert(label[..., :4].detach(), label_format, "cxcywh")
        #
        # pred_pos[..., 2:] /= 1.5
        # label[..., 2:] *= 1.5
        #
        # pred_pos = box_convert(pred_pos, "cxcywh", "xyxy")
        # area = (label[:, 2] * label[:, 3]).unsqueeze(-1)
        # label = box_convert(label, "cxcywh", "xyxy")
        # inter, union = _loss_inter_union(label.unsqueeze(1).repeat(1, pred_pos.shape[1], 1), pred_pos)
        # iou = inter / area

        inter, union = _loss_inter_union(label.unsqueeze(1).repeat(1, pred_pos.shape[1], 1), pred_pos)

        iou = inter / union

        # diou_loss = torchvision.ops.diou_loss.distance_box_iou_loss(pred_pos, label)
        # area = (label[:, 0] - label[:, 2]) * (label[:, 1] - label[:, 3])
        # area.clip_(min=0)
        # label = label.unsqueeze(1).repeat(1, pred.shape[1], 1)
        # area = area.unsqueeze(1).repeat(1, pred.shape[1])
        pred_confidences = pred[..., 4:]

        conf_all = torch.sum(pred_confidences, dim=-1)  # .clip_(min=0.001, max=1 - 0.001)
        # if conf_all >= 1, minus gap = max(conf_all - 1 + eps, 0)
        conf_all = conf_all - torch.maximum(conf_all.detach() + 1e-6 - 1, torch.zeros_like(conf_all))
        if self.supress_clsses is None:
            return self.get_loss_all_class(conf_all, inter)
        else:
            conf = torch.sum(pred_confidences[..., self.supress_clsses].float(), dim=-1)  # .clip_(min = 0.001, max = 1 - 0.001)
            conf_others = pred_confidences
            if mask is not None:
                # mask[:, self.supress_clsses] = 0
                conf_others = conf_others * mask[:, :-1].unsqueeze(1)

                # conf = pred[:, :, 4:] * (1 - mask.unsqueeze(1))
                # conf = conf.sum(dim = -1)
                conf_others = torch.cat([conf_others , ((1 - conf_all) * mask[..., -1].unsqueeze(1)).unsqueeze(-1)], dim = -1)
            else:
                # if self.tgt_classes is None:
                #     self.tgt_classes = [i for i in range(pred_confidences.shape[-1]) if (i not in self.supress_clsses)]
                conf_others = conf_others.clone()
                conf_others[..., self.supress_clsses] = 0

                conf_others = torch.cat([conf_others , ((1 - conf_all)).unsqueeze(-1)], dim=-1)

            return self.get_loss(conf, conf_all, conf_others, iou, logger, **kwargs)


    def get_loss(self, conf, conf_all, conf_others, iou, logger = None, **kwargs):
        raise NotImplementedError()


    def get_loss_all_class(self, conf_bbox, iou):
        raise NotImplementedError()


class DATKLoss(DetectionLoss):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.conf_thresh = hparams.conf_thresh
        self.iou_thresh = hparams.iou_thresh
        self.hparams = hparams

    def get_loss(self, conf, conf_all, conf_others, iou, logger = None, return_scale = True, idx_target = None):
        eps = 1e-9
        conf_mx = conf_others[..., :-1].max(dim=-1)[0] # .clip_(min=1e-10, max=1 - 0.001)

        mask = torch.logical_and(iou > self.iou_thresh, conf > self.conf_thresh)
        mask = torch.logical_and(mask, conf > conf_mx * 0.9)

        # conf_mx = conf_others.max(dim=-1)[0] # .clip_(min=1e-10, max=1 - 0.001)
        conf_mx = conf_others.sum(dim=-1) # .clip_(min=1e-10, max=1 - 0.001)

        mx_values_inter, mx_idx = torch.max(mask * iou, dim=-1)
        mx_values, mx_idx = torch.max(mask * conf, dim=-1)
        bishop = torch.logical_and(conf > mx_values.unsqueeze(-1) / 2, iou > mx_values_inter.unsqueeze(-1) / 2)
        # bishop_confuse_FP = iou < 0.1
        mask = torch.logical_and(mask, bishop)
        mask_cnt = (mask.float().sum(dim=-1, keepdim=True) + eps)
        if logger is not None:
            logger("det/mean_max_conf", mx_values.mean())
            if conf.shape[0] % 4 == 0: # TODO: extract handcraft batch idx
                block = conf.shape[0] // 4
                logger("det/max_max_conf", mx_values[block:block* 2].mean()) # max atk ratio
                logger("det/randn_max_conf", mx_values[block * 2:].mean()) # rand atk ratio
                logger("det/min_max_conf", mx_values[:block].mean()) # min atk ratio
            # print(mx_values)
            # idx = torch.stack([torch.arange(conf.shape[0]).to(mx_idx), mx_idx], dim = 1)

            # print(conf[torch.arange(conf.shape[0]).to(mx_idx), mx_idx])
            # print(conf_mx.mean())
            # print((conf_mx * mask).max(dim = -1)[0].mean())
            # exit()
            # logger("det/mask_cnt", mask)

        # IMPORTANT: due to numerical error, x may < 1. Thus, add this.  Ref: VQVAE.
        # if x <= 0, x - eps < 0, x = x - x.detach() + eps = eps, x_grad = x.
        log_adjust = lambda x: torch.log(x.float() - torch.minimum(x.float().detach() - eps , torch.zeros_like(x.detach().float())))
        # conf_others_relative = conf_others / (conf_others.sum(dim=-1, keepdims=True) + eps)
        # prob_confuse = [-log_adjust(conf_mx), (-log_adjust(conf_others / conf_mx.unsqueeze(-1)) * conf_others / conf_mx.unsqueeze(-1)).sum(dim = -1)]  # (iou).double()] # conf_others.sum(dim = -1) / conf_all.double()]
        prob_confuse = [-log_adjust(conf_mx),  (-log_adjust(conf_others) * conf_others / (conf_others.sum(dim = -1, keepdims=True) + eps)).sum(dim = -1)] # prob_confuse = [-log_adjust(1 - conf)]  # (iou).double()] # conf_others.sum(dim = -1) / conf_all.double()]
        # prob_confuse = [-log_adjust(1 - conf / conf_all)] # , (-log_adjust(conf_others) * conf_others / (conf_others.sum(dim = -1, keepdims=True) + eps)).sum(dim = -1)]  # (iou).double()] # conf_others.sum(dim = -1) / conf_all.double()]
        if idx_target is not None:
            confall_eps = conf_all.double() + eps
            if isinstance(idx_target, torch.Tensor) and len(idx_target.shape) == 1:
                arange = torch.arange(0, conf.shape[0], device=conf.device, dtype=torch.int64)
                prob_confuse.append(-log_adjust(conf[arange, ..., idx_target] / confall_eps))  # 123
            else:
                prob_confuse.append(-log_adjust(conf[..., idx_target] / confall_eps))


        logits = sum(prob_confuse)
        loss = reduction(logits * mask * iou.detach() / mask_cnt, "sum_batch")  # + reduction(- torch.log(1 - conf / conf_all) * mask)


        if not return_scale:
            return loss
        loss_value = reduction(conf * iou * mask / mask_cnt, "sum_batch").detach() + eps # +
        if torch.isnan(loss).any() or torch.isinf((loss)).any():  # all masked out
            loss = reduction(0 * conf)
            loss_value = reduction(0 * conf)

            # print(mask_cnt)
            print(torch.isnan(conf_mx).any())
            print(torch.isnan(conf_others).any())
            print(torch.isinf(conf_mx).any())
            print(torch.isinf(conf_others).any())
            print(torch.min(conf_mx))
            print(torch.min(conf_others))

            print(torch.isnan(logits).any())
            print(torch.isinf(logits).any())


            # print([(-torch.log(p + eps) * q).sum() for p, q in zip(prob_confuse, prob_select)])
            # print([p.max() for p in prob_confuse])
            # print([p.min() for p in prob_confuse])

        return loss, loss_value


def return2DGaussian(resolution, sigma, offset):
    kernel_size = resolution
    # sigma = 1.0: ± sigma covers the whole image
    sigma *= kernel_size / 2
    sigma.clip_(min = 1.0)
    offset *= kernel_size
    # https://discuss.pytorch.org/t/fast-implementation-of-gaussian-kernel/67187
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).to(sigma.device)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.unsqueeze(0)

    offset = offset.unsqueeze(1).unsqueeze(1)
    sigma = sigma.unsqueeze(1).unsqueeze(1)
    mean = (offset)
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean).float()**2. / (2*variance).float(), dim=-1))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim = (-1, -2), keepdim=True).clip_(min = 1e-6)
    return gaussian_kernel



class GeoLoss(DetectionLoss):


    def __init__(self, hparams):
        super().__init__(hparams)
        self.iou_thresh = hparams.iou_thresh

    def get_loss(self, conf, conf_all, conf_others, iou, logger=None, **kwargs):
        ret = ((conf) * (iou > self.iou_thresh)).max(dim=1)[0]
        # ret = ((conf) * (iou > self.iou_thresh) * (conf > 0.3)).sum(dim=1) / ((iou > self.iou_thresh) * (conf > 0.3)).sum(dim = 1).clip(min = 1)
        return ret, torch.ones_like(ret)


class RevGeoLoss(DetectionLoss):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.iou_thresh = hparams.iou_thresh

    def get_loss(self, conf, conf_all, conf_others, iou, logger=None, **kwargs):
        ret = ((-conf) * (iou > self.iou_thresh)).max(dim=1)[0]
        return ret, torch.ones_like(ret)







if __name__ == '__main__':
    # backbone = torchvision.models.resnet50(pretrained = True).cuda()
    # x = torch.randn(16, 3, 224, 224, device="cuda")
    # x1 = torch.randn(16, 3, 224, 224, device="cuda")
    # out = backbone(x)
    # out1 = backbone(x1)
    # cr = CrossEntropy()
    # print(cr(out1, out))
    # print(cr(out, out))
    bbox0 = [[[100, 100, 300, 300, 0.99, 0.99], [100, 100, 300, 300, 0.99, 0.99]]]
    bbox1 = [[195, 195, 202, 202]]
    loss1 = DATKLoss(target_clsses = [0])
    print(loss1(torch.tensor(bbox0),torch.tensor( bbox1), return_scale = True))
