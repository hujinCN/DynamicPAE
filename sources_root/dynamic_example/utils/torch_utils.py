import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn as nn
import numpy as np
import torchvision
def bf16_grid_sample(input: Tensor, grid: Tensor, *args, **kwargs):
    input_dtype = input.dtype
    if input_dtype == torch.bfloat16:
        input = input.float()
    out = grid_sample(input, grid, *args, **kwargs)
    if input_dtype == torch.bfloat16:
        out = out.to(torch.bfloat16)
    return out

if F.grid_sample != bf16_grid_sample:
    grid_sample = F.grid_sample
    F.grid_sample = bf16_grid_sample

interpolate = F.interpolate


def bf16_interpolate(input: Tensor, *args, **kwargs):
    input_dtype = input.dtype
    if input_dtype == torch.bfloat16:
        input = input.float()
    out = interpolate(input, *args, **kwargs)
    if input_dtype == torch.bfloat16:
        out = out.to(torch.bfloat16)
    return out


F.interpolate = bf16_interpolate


def bf16_grid_sampler(input: Tensor, grid: Tensor, *args, **kwargs):
    input_dtype = input.dtype
    if input_dtype == torch.bfloat16:
        input = input.float()
    out = grid_sampler(input, grid, *args, **kwargs)
    if input_dtype == torch.bfloat16:
        out = out.to(torch.bfloat16)
    return out

if torch.grid_sampler != bf16_grid_sampler:
    grid_sampler = torch.grid_sampler
    torch.grid_sampler = bf16_grid_sampler



def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._NormBase):
            module.eval()
            module.train = lambda x : x
            module.skip_sync_bn_convert = True
            if hasattr(module, "weight"):
                module.weight.requires_grad = False
            if hasattr(module, "bias"):
                module.bias.requires_grad = False
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()
            module.train = lambda x : x


def convert_sync_bn(model, process_group = None, gpu=None):
    # convert all BN layers in the model to syncBN
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if not hasattr(child, "skip_sync_bn_convert"):
                m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
                if (gpu is not None):
                    m = m.cuda(gpu)
                setattr(model, child_name, m)
            else:
                print("skipped a synbn convert")
        else:
            convert_sync_bn(child, process_group, gpu)
    return model
def inter_nms(all_predictions, conf_thres=0.25, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    max_det = 300  # maximum number of detections per image
    out = []
    for predictions in all_predictions:
        # for each img in batch
        # print('pred', predictions.shape)
        if not predictions.shape[0]:
            out.append(predictions)
            continue
        if type(predictions) is np.ndarray:
            predictions = torch.from_numpy(predictions)
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        # print(predictions.shape[0])
        scores = predictions[:, 4]
        i = scores > conf_thres

        # filter with conf threshhold
        boxes = predictions[i, :4]
        scores = scores[i]

        # filter with iou threshhold
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # print('i', predictions[i].shape)
        out.append(predictions[i])
    return out


class freeze_rnd_state:
    def __enter__(self) -> None:
        self.torch_state = torch.get_rng_state()
        self.pyrnd_state = random.getstate()
        self.np_rnd_state = np.random.get_state()


    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_rng_state(self.torch_state)
        random.setstate(self.pyrnd_state)
        np.random.set_state(self.np_rnd_state)




