import itertools
from typing import Union, List

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from dynamic_example.models.prob_models import MotionSimulator


# def param2theta(param, w, h):
#     param = np.linalg.inv(param)
#     theta = np.zeros([2,3])
#     theta[0,0] = param[0,0]
#     theta[0,1] = param[0,1]*h/w
#     theta[0,2] = param[0,2]*2/w + param[0,0] + param[0,1] - 1
#     theta[1,0] = param[1,0]*w/h
#     theta[1,1] = param[1,1]
#     theta[1,2] = param[1,2]*2/h + param[1,0] + param[1,1] - 1
#     return theta


class ApplyPatchToBBox(torch.nn.Module):
    def __init__(self, scale=0.2):
        super().__init__()
        self._scale = scale
        self._rotate = None
        self.register_buffer("affine_mat", None, persistent=False)
        self.register_buffer("affine_mat_inv", None, persistent=False)

    @property
    def scale(self):
        return self._scale

    @property
    def rotate(self):
        return self._rotate * (torch.rand(self.cur_bs) - .5)

    @property
    def do_rotate(self):
        return self._rotate != None

    def apply_patch(self, img, patch, ret_mask):
        w1, h1 = patch.shape[-2], patch.shape[-1]
        w1, h1 = min(w1, img.shape[-2]), min(h1, img.shape[-1])

        patch = patch[:, :, :w1, :h1]
        mask = patch < 1
        if ret_mask:
            return torch.where(mask, img, patch - 1), mask
        return torch.where(mask, img, patch - 1)

    def update_affine_matrix(self, bboxes: torch.Tensor, patch_c, patch_h, patch_w, bg_h, bg_w):  # ltwh
        batch_size, dat = bboxes.shape
        self.cur_bs = batch_size
        device = bboxes.device
        self.cur_device = device

        # resize
        box_h, box_w = bboxes[:, 3], bboxes[:, 2]
        box_hc, box_wc = bboxes[:, 1] + box_h / 2, bboxes[:, 0] + box_w / 2
        box_hc /= bg_h
        box_wc /= bg_w

        # img_size = (bg_h **0.5+ bg_w **0.5) **2
        current_scale = self.scale
        patch_ratio = current_scale * torch.sqrt(box_w ** 2 + box_h ** 2) / math.sqrt(bg_w ** 2 + bg_h ** 2)

        patch_size = math.sqrt(patch_h ** 2 + patch_w ** 2)

        self.update_affine_mat(batch_size, patch_c, bg_h, bg_w, box_hc, box_wc, patch_h, patch_w, patch_ratio, device)

    def update_affine_mat__same_shape(self, bs, c, h, w, loc_h, loc_w, mask_ratio, device):
        return self.update_affine_mat(bs, c, h, w, loc_h, loc_w, h, w, mask_ratio, device)

    def update_affine_mat_transform(self, ccwht, batch_template, background_template, update_inv = True, update_mat = True):
        batch_size,  channel_size,  patch_original_h, patch_original_w= batch_template.shape
        _, _, bg_h, bg_w = background_template.shape
        device = ccwht.device
        last_dim = ccwht.shape[-1]

        ratio0 = bg_h / patch_original_h
        ratio1 = bg_w / patch_original_w
        real_ratio = max(ratio1, ratio0)
        out_patch_h = math.ceil(real_ratio * patch_original_h)
        out_patch_w = math.ceil(real_ratio * patch_original_w)
        coord_sclae_h = real_ratio / ratio0
        coord_scale_w = real_ratio / ratio1

        theta = torch.zeros((batch_size, 3, 3), device=device).float()
        theta_inv = torch.zeros((batch_size, 3, 3), device=device).float()
        theta[:, 2, 2] = theta_inv[:, 2, 2] = 1
        if last_dim == 4:
            x, y, w, h = [i.squeeze(-1) for i in torch.split(ccwht, 1, dim = -1)]
            theta[:, 0, 0], theta[:, 1, 1] = 1, 1
            theta_inv[:, 0, 0], theta_inv[:, 1, 1] = 1, 1
            x, y = 1 - 2 * x / coord_scale_w, 1 - 2 * y  / coord_sclae_h
            theta[:, 0, 2], theta[:, 1, 2] = x, y
            theta_inv[:, 0, 2], theta_inv[:, 1, 2] = -x, -y
        else:
            scale_x = 1
            scale_y = 1
            if last_dim == 5:
                x, y, w, h, t = [i.squeeze(-1) for i in torch.split(ccwht, 1, dim = -1)]
            else:
                assert last_dim == 7, "Unsupported format"
                x, y, w, h, t, rx, ry = [i.squeeze(-1) for i in torch.split(ccwht, 1, dim = -1)]
                scale_x = 1 / torch.cos(rx).to(device)
                scale_y = 1 / torch.cos(ry).to(device)

            x, y = 1 - 2 * x / coord_scale_w, 1 - 2 * y  / coord_sclae_h
            sin = torch.sin(t).to(device)
            cos = torch.cos(t).to(device)

            theta[:, 0, 0], theta[:, 0, 1] =  cos * scale_x, sin * scale_y
            theta[:, 1, 0], theta[:, 1, 1] = -sin * scale_x, cos * scale_y

            theta_inv[:, 0, 0], theta_inv[:, 0, 1] =  cos / scale_x, -sin / scale_x
            theta_inv[:, 1, 0], theta_inv[:, 1, 1] =  sin / scale_y, cos / scale_y

            theta[:, 0, 2] =  x * cos * scale_x + y * sin * scale_y
            theta[:, 1, 2] = -x * sin * scale_x + y * cos * scale_y

            theta_inv[:, 0, 2] = -x# -x * cos / scale_x + y * sin / scale_y
            theta_inv[:, 1, 2] = -y# -x * sin / scale_x - y * cos / scale_y

        theta[:, 0] /= w.unsqueeze(-1)
        theta[:, 1] /= h.unsqueeze(-1)
        # theta_inv[:, 0, :2] *= w.unsqueeze(-1)
        # theta_inv[:, 1, :2] *= h.unsqueeze(-1)
        theta_inv[:, :, 0] *= w.unsqueeze(-1)
        theta_inv[:, :, 1] *= h.unsqueeze(-1)

        # print(torch.bmm(theta, theta_inv))

        self.sample_length = torch.linalg.eigvals(theta[:, :2, :2]).abs() / (out_patch_h ** 2 + out_patch_w ** 2) ** 0.5 * (patch_original_h ** 2 + patch_original_w ** 2) ** 0.5


        if update_mat:
            self.affine_mat = F.affine_grid(theta[:, :2], (batch_size, channel_size, int(out_patch_h), int(out_patch_w)),
                                        align_corners=False).float()

        if update_inv:
            self.affine_mat_inv = F.affine_grid(theta_inv[:, :2], (batch_size, channel_size, int(patch_original_h), int(patch_original_w)),
                                        align_corners=False).float()

    def reverse(self):
        self.affine_mat, self.affine_mat_inv = self.affine_mat_inv, self.affine_mat
    def update_affine_mat(self, batch_size, channle_size, bg_h, bg_w, loc_h, loc_w, patch_original_h, patch_original_w,
                          mask_ratio, device):
        ratio0 = bg_h / patch_original_h
        ratio1 = bg_w / patch_original_w
        real_ratio = max(ratio1, ratio0)
        out_patch_h = math.ceil(real_ratio * patch_original_h)
        out_patch_w = math.ceil(real_ratio * patch_original_w)
        coord_sclae_h = real_ratio / ratio0
        coord_scale_w = real_ratio / ratio1


        loc_h, loc_w = loc_h / coord_sclae_h, loc_w / coord_scale_w
        theta = torch.zeros((batch_size, 2, 3), device=device).float()
        theta_inv = torch.zeros((batch_size, 2, 3), device=device).float()

        loc_h, loc_w = 1 - 2 * loc_h, 1 - 2 * loc_w
        if isinstance(mask_ratio, torch.Tensor):
            mask_ratio = mask_ratio.reshape(-1, 1, 1)
        if self.do_rotate:
            rotate = self.rotate
            sin = torch.sin(rotate).to(device)
            cos = torch.cos(rotate).to(device)

            theta[:, 0, 0] = cos
            theta[:, 0, 1] = sin
            theta[:, 0, 2] = loc_w * cos + loc_h * sin
            theta[:, 1, 0] = -sin
            theta[:, 1, 1] = cos
            theta[:, 1, 2] = -loc_w * sin + loc_h * cos

            theta /= mask_ratio
            self.affine_mat = F.affine_grid(theta, (batch_size, channle_size, int(out_patch_h), int(out_patch_w)),
                                            align_corners=False)
            self.affine_mat_inv = None
        else:
            theta[:, 0, 0], theta[:, 0, 1],theta[:, 0, 2]  = 1, 0, loc_w
            theta[:, 1, 0], theta[:, 1, 1],theta[:, 1, 2]  = 0, 1, loc_h
            theta /= mask_ratio
            theta_inv[:, 0, 0], theta_inv[:, 0, 1],theta_inv[:, 0, 2]  = 1, 0, -loc_w
            theta_inv[:, 1, 0], theta_inv[:, 1, 1],theta_inv[:, 1, 2]  = 0, 1, -loc_h
            theta_inv[:, [0], :2] *= mask_ratio
            theta_inv[:, [1], :2] *= mask_ratio
            # print(theta)
            # print(theta_inv)

            self.affine_mat = F.affine_grid(theta, (batch_size, channle_size, int(out_patch_h), int(out_patch_w)),
                                            align_corners=False).float()
            self.affine_mat_inv = F.affine_grid(theta_inv, (batch_size, channle_size, int(patch_original_h), int(patch_original_w)),
                                            align_corners=False).float()



    def forward(self, backgronds: torch.Tensor, patches: torch.Tensor, return_mask=False, interpolate_grad = False):
        # t_patches = F.grid_sample(patches + 1, self.affine_mat, padding_mode="zeros", mode='nearest')
        if self.affine_mat_inv is not None and patches.requires_grad and interpolate_grad:
            # assert return_mask is False
            ans = GridSample.apply(patches,  self.affine_mat, self.affine_mat_inv, backgronds, torch.tensor(False))
            # ans = self.apply_patch(backgronds, ans, return_mask)
            return ans
        else:
            t_patches = F.grid_sample(patches + 1, self.affine_mat, padding_mode="zeros", mode="bilinear", align_corners = False)
            return self.apply_patch(backgronds, t_patches, return_mask)

    def get_sample_length(self):
        return self.sample_length




class GridSample(torch.autograd.Function):
    """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """

    @staticmethod
    def forward(ctx, patch, affine, affine_inv, backgrounds, return_mask):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        patch = F.grid_sample(patch + 1, affine)
        w1, h1 = patch.shape[-2], patch.shape[-1]
        ctx.save_for_backward(affine_inv, torch.tensor(w1),  torch.tensor(h1))
        w1, h1 = min(w1, backgrounds.shape[-2]), min(h1, backgrounds.shape[-1])
        # return patch
        patch = patch[:, :, :w1, :h1].contiguous()

        mask = patch < 1
        if return_mask:
            return torch.where(mask, backgrounds, patch - 1), mask
        return torch.where(mask, backgrounds, patch - 1)

    @staticmethod
    def backward(*args):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        ctx = args[0]
        grad_output = args[1]
        affine_inv, w1, h1 = ctx.saved_tensors
        w1, h1 = int(w1), int(h1)
        if w1 != grad_output.shape[-1] or h1 != grad_output.shape[-2]:
            new_t = torch.zeros(grad_output.shape[0], grad_output.shape[1], h1, w1, device = grad_output.device, dtype=grad_output.dtype)
            new_t[:, :, :grad_output.shape[-2], :grad_output.shape[-1]] = grad_output
            grad_output = new_t
        return F.grid_sample(grad_output, affine_inv), None, None, None, None


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape: Union[tuple | List[int]], scaleup=True, center=True, transform_aug=None, norm_bbox = True, border = cv2.BORDER_CONSTANT):
        """Initialize LetterBox object with specific parameters. new_shape: (h, w)"""
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.center = center  # Put the image in the middle or top-left
        self.img_trans = transform_aug
        self.norm_bbox = norm_bbox
        self.border = border

    def __call__(self, img, target = None, return_params  = False, multiple_tgt = False):
        # img = labels.get('img') if image is None else image
        if self.img_trans is not None:
            img = self.img_trans(img)

        img = np.asarray(img)

        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # w, h
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, self.border,
                                 value=(114, 114, 114))  # add border
        ret = [torchvision.transforms.ToTensor()(img)]
        if target is not None:
            ret.append(self.update_labels(torch.as_tensor(target), r, dh, dw))
        if return_params:
            ret.append((r, dh, dw))
        return ret[0] if len(ret) == 0 else tuple(ret)


    def update_labels(self, bboxes, scale, dh, dw):  # transfrom in ccwh
        bboxes = torch.clone(bboxes)
        if(bboxes.shape[0] == 0):
            return bboxes
        bboxes *= scale
        bboxes[..., 0] += dw
        bboxes[..., 1] += dh
        if self.norm_bbox:
            bboxes[..., [0, 2]] /= self.new_shape[1] # w
            bboxes[..., [1, 3]] /= self.new_shape[0] # h
        return bboxes

class CamTrans:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640)):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        # self.scaleup = scaleup
        # self.center = center  # Put the image in the middle or top-left
        # self.img_trans = transform_aug

    def __call__(self, img, ccwh, gt = None):
        """
        :param img: img.shape[-2:] ~ self.new_shape
        """
        x, y, w, h = [i.squeeze(-1) for i in torch.split(ccwh, 1, dim = -1)]

        bs = x.shape[0]
        scale_up = torch.min(self.new_shape[0] / w / img.shape[-1], self.new_shape[1] / h / img.shape[-2])
        # box_hc, box_wc = box_hc / img_max_size, box_wc / img_max_size
        # coord_sclae_h = img_max_size / x.shape[-2]
        # coord_scale_w = img_max_size / x.shape[-1]
        bs = x.shape[0]
        # target_size = torch.sqrt(box_w ** 2 + box_h ** 2)  # .reshape(bs, 1, 1)
        # target_size = target_size / (x.shape[-2] ** 2 + x.shape[-1] ** 2) ** 0.5 * self.input_field
        l, t = x - w / 2, y - h / 2
        x, y = - 1 + 2 * x, - 1 + 2 * y
        theta = torch.zeros((bs, 2, 3), device=x.device)
        theta[:, 0, 0] = 1 / scale_up
        theta[:, 0, 1] = 0
        theta[:, 0, 2] = x
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = 1 / scale_up
        theta[:, 1, 2] = y
        affine_grid = F.affine_grid(theta, (bs, img.shape[1], self.new_shape[0], self.new_shape[1]), False)
        input = F.grid_sample(img, affine_grid)

        if gt is None:
            return input


        if isinstance(gt, list):
            return input, [self.update_gt(scale_up, l, t, g) for g in gt]
        return input, self.update_gt(scale_up, l, t, gt)

    def update_gt(self, scale_up, l, t, gt):
        x1 = torch.clone(gt)
        x1[:, 0] -= l
        x1[:, 1] -= t
        x1[:, :4] *= scale_up.unsqueeze(-1)
        return x1


def __apply_bbox__(img, ltwh):
    img[..., ltwh[..., 1].long():(ltwh[..., 1] + ltwh[..., 3]).long(),
    ltwh[..., 0].long():(ltwh[..., 0] + ltwh[..., 2]).long()] = 0




def __test__():
    img1 = torch.ones(1, 3, 224, 448)
    img2 = torch.ones(1, 3, 124, 124) * 0.5
    img2[:, :, 4:66, 4:66] *= 0.5
    img2.requires_grad = True
    apply = ApplyPatchToBBox()
    apply.update_affine_matrix(torch.tensor([[50, 50, 100, 100]]), 3, 224, 224, 224, 448)
    img3 = apply(img1, img2)**2
    plt.imshow(img3[0].detach().permute(1, 2, 0).numpy())
    plt.show()
    loss = img3.sum()
    loss.backward()
    plt.imshow(img2.grad.data[0].permute(1, 2, 0).numpy())
    plt.show()
    img1 = torch.ones(1, 3, 480, 640)
    img2 = torch.ones(1, 3, 224, 224) * 0.5
    img2[:, :, 4:112, 4:112] *= 0.5
    img2.requires_grad = True
    transform = MotionSimulator()
    tt = transform.sample(torch.zeros(1, 7))
    tt[[0]] = torch.tensor([ 0.3845,  0.5144,  0.0317,  0.0829, -1.2649,  0.7208,  0.3447])
    # tt[[0]] = torch.tensor([ 0.3845,  0.5144,  0.0317,  0.0329])
    print(tt)
    apply.update_affine_mat_transform(tt, img2, img1)
    img3 = apply(img1, img2) ** 2
    loss = img3.sum()
    loss.backward()
    plt.imshow(img3[0].detach().permute(1, 2, 0).numpy())
    plt.show()

    plt.imshow(img2.grad.data[0].permute(1, 2, 0).numpy())
    plt.show()



if __name__ == '__main__':
    __test__()

class RandLocalTrans(torch.nn.Module):
    """
    Implemented with torchvision.transforms.ColorJitter.
    See Also: Athalye, A, et, al. "Synthesizing Robust Adversarial Examples." (EoT)
    """

    def __init__(self,
                 brightness=0.1,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 gaussian_sigma=None,  # (0.1, 1)
                 kernel_sz=int(5),
                 noise_sigma=0.1):
        super(RandLocalTrans, self).__init__()
        self.augs = []
        if brightness is not None or contrast is not None or saturation is not None or hue is not None:
            if brightness is None:
                brightness = 0
            if contrast is None:
                contrast = 0
            if saturation is None:
                saturation = 0
            if hue is None:
                hue = 0
            aug = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                                                     hue=hue)
            self.augs.append(aug)
        if gaussian_sigma is not None:
            self.augs.append(torchvision.transforms.GaussianBlur(kernel_size=kernel_sz, sigma=gaussian_sigma))
        self.augs = torch.nn.Sequential(*self.augs)
        self.noise_sigma = noise_sigma

    def forward(self, img):
        img = self.augs(img)
        if self.noise_sigma is not None:
            img = torch.clamp(img + self.noise_sigma * torch.randn_like(img), 0.0, 1.0)
        return img


class RandTPS(nn.Module):
    """
    Random Thin-Plate Spine Aug.
    Sources:
    https:github.com/WarBean/tps_stn_pytorch/blob/master/tps_grid_gen.py
    https://github.com/WhoTHU/Adversarial_Texture/blob/master/tps_grid_gen.py
    """

    @staticmethod
    def compute_partial_repr(input_points, control_points):

        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        # @WarBean
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def __init__(self, target_shape, target_control_points):
        super(RandTPS, self).__init__()
        target_height, target_width = target_shape
        self.target_shape = target_shape

        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # self.target_control_points = target_control_points
        self.register_buffer('target_control_points', target_control_points)

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = RandTPS.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = RandTPS.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    @staticmethod
    def grid_sample(input, grid, canvas=None):
        output = F.grid_sample(input, grid)
        if canvas is None:
            return output
        else:
            input_mask = Variable(input.data.new(input.size()).fill_(1))
            output_mask = F.grid_sample(input_mask, grid)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output

    def get_source_grids(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate

    def forward(self, inputs, source_control_points=None, max_range=0.1, canvas=0.5, target_shape=None):
        if target_shape is not None:
            if target_shape != self.target_shape:
                device = self.padding_matrix.device
                self.__init__(target_shape, self.target_control_points.cpu())
                self.to(device)

        target_height, target_width = self.target_shape
        # @WhoThu Rand. Sampling.
        if source_control_points is None:
            source_control_points = self.target_control_points.unsqueeze(0) + self.target_control_points.new(
                size=(inputs.shape[0],) + self.target_control_points.shape).uniform_(-max_range, max_range)
            # source_control_points = source_control_points.to(self.padding_matrix.device)
        if isinstance(canvas, float):
            canvas = torch.FloatTensor(inputs.shape[0], 3, target_height, target_width).fill_(canvas).to(
                self.padding_matrix.device)
        source_coordinate = self.get_source_grids(source_control_points)
        grid = source_coordinate.view(inputs.shape[0], target_height, target_width, 2)
        target_image = RandTPS.grid_sample(inputs, grid, canvas)
        return target_image, source_control_points

def compose_boxes(ccwh_trans0, ccwhrrr_trans1):
        ccwhrrr_patch_0 = torch.clone(ccwhrrr_trans1)
        ccwhrrr_patch_0[:, :2] *= ccwh_trans0[:, 2:4]
        ccwhrrr_patch_0[:, 2:4] *= ccwh_trans0[:, 2:4]
        ccwhrrr_patch_0[:, :2] += ccwh_trans0[:, :2] - ccwh_trans0[:, 2:4] / 2
        return ccwhrrr_patch_0