import math
from typing import List, Optional

import torch
import torch.nn
from torch.distributions import Exponential, Uniform, Normal
from torchvision.transforms import ColorJitter


class TaskRatioSampler(torch.nn.Module):
    def __init__(self, dim=2, positive_dims=None):
        super().__init__()
        if positive_dims is None:
            positive_dims = [0, 1]
        self.dim = dim
        self.positive_dims = positive_dims
        self.val_ratio = None

    def forward(self, template):
        bs = template.shape[0]

        if self.val_ratio is not None:
            v = torch.tensor(self.val_ratio, device=template.device, dtype=torch.float32).unsqueeze(0).repeat(bs, 1)
        else:
            assert bs % 4 == 0
            block = bs // 4
            v = torch.randn(bs,  self.dim, device=template.device, dtype=torch.float32)
            v[:block, 0] = v[block:block * 2, 1] = 0
            v[:block, 1] = v[block:block * 2, 0] = 1

        norm = torch.linalg.norm(v, ord=1, dim = 1, keepdim=True).clip(min=1e-8)
        v = v / norm
        v[:, self.positive_dims] = v[:, self.positive_dims].abs()
        return v


    def set_val_ratio(self, ratio: List[float]):
        self.val_ratio = ratio

    def get_atk_batches(self, task_ratio):
        bs = task_ratio.shape[0]
        assert bs % 4 == 0
        block = bs // 4
        range = torch.arange(block, block * 2, device=task_ratio.device)
        return range
        # return task_ratio[:, 0] > task_ratio[:, 1]



class TaskWeightBalancing(torch.nn.Module):
    def __init__(self, num_task = 2, target_skewness_ratio=None):
        super().__init__()
        assert num_task >= 2
        if target_skewness_ratio is None:
            target_skewness_ratio = [0 for _ in range(num_task)]
        self.num_task = num_task
        self.target_skewness_ratio = target_skewness_ratio
        self._ratio = torch.nn.Parameter(torch.zeros(num_task - 1))
        self.norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(1) for _ in range(num_task)])
        self.norms3 = torch.nn.ModuleList([torch.nn.BatchNorm1d(1) for _ in range(num_task)])
        self.norms_minmax = torch.nn.ModuleList([torch.nn.BatchNorm1d(1) for _ in range(num_task)])
    @property
    def ratio(self):
        return self._ratio.exp().clamp(min = 1e-6, max = 1e6)
    def forward(self, loss_min_max_sampled, num_tsk: int):
        if torch.isnan(loss_min_max_sampled).any() or torch.isinf(loss_min_max_sampled).any():
            return
        bs = loss_min_max_sampled.shape[0]
        assert bs % 4 == 0
        block = bs // 4
        loss_min = loss_min_max_sampled[:block]
        loss_max = loss_min_max_sampled[block:block * 2]
        loss_sampled = loss_min_max_sampled[block * 2: ]
        self.norms[num_tsk](loss_sampled.reshape(-1, 1))
        self.norms3[num_tsk]((loss_sampled.reshape(-1, 1) - self.norms_minmax[num_tsk].running_mean) ** 3)
        self.norms_minmax[num_tsk]((loss_min + loss_max).reshape(-1, 1) / 2)
        # if num_tsk == 0:
        #     return loss_min + loss_max + loss_sampled * 2
        # else:
        #     return self.ratio[num_tsk - 1].detach() * (loss_min + loss_max + loss_sampled * 2)

    def loss(self):
        ratio_skewness0 = self.get_skewness(0) - self.target_skewness_ratio[0]

        loss = 0
        for i in range(1, self.num_task):
            ratio_skewness = self.get_skewness(i)  - self.target_skewness_ratio[i]
            # sk + log - loss - grad_param + ratio ++
            loss += self.ratio[i - 1] * ( - ratio_skewness).detach() # gradient = torch.log(ratio_skewness / ratio_skewness0)
        return loss

    def get_skewness(self, i: int):
        ratio_skewness0 = self.norms3[i].running_mean.float() / (self.norms[i].running_var.clip(min=1e-7) ** 1.5).float()
        return ratio_skewness0


# The order of lists corresponds to the ratio from TaskRatioSampler.
# def interpolate_dist(mu: List[torch.Tensor], sigma: List[torch.Tensor], ratio: torch.Tensor):
#     dim_ratio = ratio.shape[1]
#     assert len(mu) == len(sigma) and len(mu) == dim_ratio
#     dim_ratio = int(dim_ratio)
#     mu1 = sum([mu[i] * ratio[:, [i], None, None] for i in range(dim_ratio)])
#     sigma1 = sum([sigma[i] * ratio[:, [i], None, None].abs() for i in range(dim_ratio)])
#     eps = torch.randn_like(mu1)
#     return mu1 + sigma1 * eps

# def interpolate_dist(mu: List[torch.Tensor], sigma: List[torch.Tensor], ratio: torch.Tensor):
#     bs, _, h, w = mu[0].shape
#
#     theta = torch.atan2(ratio[:, 0], ratio[:, 1])
#     z0 = torch.randn_like(mu[0]) * sigma[0] + mu[0]
#     z1 = torch.randn_like(mu[1]) * sigma[1] + mu[1]
#     return torch.cat([z0 * theta[:, None, None, None], z1, theta[:, None, None, None].repeat(1, 1, h, w)], dim = 1)

# todo : p(z | z_hide, z_patch)

def interpolate_dist(mu: List[torch.Tensor], sigma: List[torch.Tensor], ratio: torch.Tensor, training = True):
    bs, _, h, w = mu[0].shape

    # theta = torch.atan2(ratio[:, 0], ratio[:, 1])
    def inter_1(x, y):
        ones = torch.ones_like(x)
        return x * y + (1 - y) * ones
    # z0 = torch.randn_like(mu[0]) * inter_1(sigma[0], ratio[:, 0, None, None, None]) + mu[0] * ratio[:, 0, None, None, None]
    # z1 = torch.randn_like(mu[1]) * inter_1(sigma[1], ratio[:, 1, None, None, None]) + mu[1] * ratio[:, 1, None, None, None]

    if training:
        z1 = torch.randn_like(mu[1]).clip(min = -100, max = 100) * sigma[1]  + mu[1]
        z0 = torch.randn_like(mu[0]).clip(min = -100, max = 100) * sigma[0]  + mu[0]
    else:
        z1 = mu[1]
        z0 = mu[0]
    # z0 = mu[0]
    # z1 = mu[1]
    # ratio_sum = ratio[:, 0, None, None, None] + ratio[:, 1, None, None, None]
    return torch.cat([z0 , z1], dim = 1)











if __name__ == '__main__':
    a = TaskRatioSampler()
    b = torch.ones(30,3)
    print(a(b).float())



class ColorSimulator(ColorJitter):
    def __init__(self,
                 brightness = 0.05, saturation = 0.1, contrast = 0.1,  gamma = 0.0, blur = 0.0, noise = 0.02,
                 t_brightness = 0.02, t_saturation = 0.01, t_contrast = 0.01,  t_gamma = 0.01, t_blur = 0.00, t_noise = 0.002):
        super().__init__()
        self.sample_params = [brightness, contrast, saturation, gamma, blur,  noise]
        self.adjust_params = [t_brightness, t_contrast, t_saturation,t_gamma, t_blur, t_noise]

    def sample(self, template_bs):
        template_bs = torch.zeros(template_bs.shape[0]).to(template_bs)
        b, s, c, g = [torch.empty_like(template_bs).uniform_(1 - i, 1 + i) for i in self.sample_params[:4]]
        bb, nn = self.sample_params[4:]
        blur = torch.empty_like(template_bs).uniform_(0, bb)
        noise = torch.empty_like(template_bs).uniform_(0, nn)
        return torch.stack([b, s, c, g, blur, noise], dim = -1)


    def update(self, anchor_bcsg_gauss_noise):
        scale = torch.tensor(self.adjust_params).to(anchor_bcsg_gauss_noise).unsqueeze(0)
        return anchor_bcsg_gauss_noise + scale * torch.randn_like(anchor_bcsg_gauss_noise)


class MotionSimulator:
    """
    Define a format of patch transformation.
    [center_position_x, center_position_y, (relative)width, (relative)height, ]
    """
    def __init__(self, pic_size=1.0,
                 pos_mu=(0.5, 0.5), size_mu=(0.2, 0.2), size_min = None,
                 pos_range=0.1, size_range=0.01, rotate_z_sigma = 0.125, rotate_sigma = 0.125,
                 incremental = 0.1, force_ratio = None):
        self.pic_size = pic_size
        self.pos_range = pos_range
        self.size_range = size_range
        self.rotate_z_sigma = rotate_z_sigma
        self.size_mu, self.pos_mu = size_mu, pos_mu
        # self.pos_distribution = Exponential(rate=1 / pos_sigma)
        self.size_min = size_min
        self.force_ratio = force_ratio
        self.rotate_sigma = rotate_sigma
        self.incremental = incremental

    def sample(self, ccwht, sample_around = True, dim_trans = 7) -> torch.Tensor:
        template = ccwht[:, 0]
        # noinspection PyTypeChecker
        # theta_pos: torch.Tensor = torch.rand_like(template) * (math.pi * 2)  # [0, 2pi]
        # theta_ = torch.rand_like(template) * (math.pi * 2)  # [0, 2pi]

        # clip for range μ ± μ / 2
        w = (torch.rand_like(template) - 0.5) * 2 * self.size_range + self.size_mu[0]

        if self.force_ratio is not None:
            h = w.clone() * self.force_ratio
        else:
            w = (torch.rand_like(template) - 0.5) * 2 * self.size_range + self.size_mu[0]
            h = (torch.rand_like(template) - 0.5) * 2 * self.size_range + self.size_mu[1]


        x = (torch.rand_like(template) - 0.5) * 2 * self.pos_range
        y = (torch.rand_like(template) - 0.5) * 2 * self.pos_range

        if sample_around: # area ratio
            x = x * ccwht[:, 2] + ccwht[:, 0]
            y = y * ccwht[:, 3] + ccwht[:, 1]

            s = (ccwht[:, 2] * ccwht[:, 3])
            if self.force_ratio == 1.0:
                w = (w * s) ** 0.5
                h = w.clone()
            else:
                raise NotImplementedError()
            if self.size_min is not None:
                w = torch.clip(w, min=self.size_min[0])
                h = torch.clip(h, min=self.size_min[1])
        else:
            x += self.pos_mu[0]
            y += self.pos_mu[1]


            # x = d * torch.cos(theta_pos) + self.pos_mu[0]
        # x.clip_(min = 0, max = 1)
        # y = d * torch.sin(theta_pos) + self.pos_mu[1]
        # y.clip_(min = 0, max = 1)


        if dim_trans == 4:
            # ccwh
            return torch.stack([x, y, w, h], dim = -1) * self.pic_size
        elif dim_trans == 5:
            # ccwht
            # noinspection PyTypeChecker
            theta_rotate: torch.Tensor = torch.rand_like(template) * (math.pi * 2)- math.pi  # [0, 2pi]
            theta_rotate *= self.rotate_sigma
            return torch.stack([x, y, w, h, theta_rotate], dim = -1) * self.pic_size

        else:
            assert dim_trans == 7, "Unsupport format"
            # ccwh rrr
            # noinspection PyTypeChecker
            theta_rotate: torch.Tensor = torch.rand_like(template) * (math.pi * 2) - math.pi  # [0, 2pi]
            theta_rotate *= self.rotate_sigma
            # noinspection PyTypeChecker
            theta_rotate_x: torch.Tensor = (torch.rand_like(template) - .5) * 2 * self.rotate_z_sigma * math.pi / 2# μ = 0
            # noinspection PyTypeChecker
            theta_rotate_y: torch.Tensor = (torch.rand_like(template) - .5) * 2 * self.rotate_z_sigma * math.pi / 2# μ = 0
            return torch.stack([x, y, w, h, theta_rotate,
                                theta_rotate_x.clip_(min = - math.pi / 2, max = math.pi / 2),
                                theta_rotate_y.clip_(min = - math.pi / 2, max = math.pi / 2)], dim = -1) * self.pic_size

    def sample_incre(self,  cxcywhttt, ccwht, sample_around = True, dim_trans = 7):
        return self.sample(ccwht, sample_around, dim_trans) * (1 - self.incremental) + cxcywhttt * self.incremental
    # def move(self, ccwht, inplace = False, relative = False) -> torch.Tensor:
    #     bs = ccwht.shape[0]
    #     template = ccwht[:, 0]
    #     d = self.v_xy_distribution.sample(template.shape).to(template)
    #     if relative:
    #         dmin = torch.min( ccwht[:, 2], ccwht[:, 3])
    #         d = d * dmin
    #
    #     # noinspection PyTypeChecker
    #     theta_pos: torch.Tensor = torch.rand_like(template) * (math.pi * 2)  # [0, 2pi]
    #     x = d * torch.cos(theta_pos)
    #     y = d * torch.sin(theta_pos)
    #
    #     cw, ch = self.size_mu[0]  / 2, \
    #              self.size_mu[1]  / 2
    #     # noinspection PyUnresolvedReferences
    #     w = (self.v_scale_distribution.sample(torch.Size([bs])).clip(min=0, max=cw) * (torch.randint(0,2,[bs]) * 2 - 1) ).to(template)
    #     if self.force_ratio:
    #         h = w.clone()
    #     else:
    #         h = (self.v_scale_distribution.sample(torch.Size([bs])).clip(min=0, max=ch) * (torch.randint(0,2,[bs]) * 2 - 1)).to(template)
    #
    #
    #     if relative:
    #         w = w * ccwht[:, 2]
    #         h = h * ccwht[:, 3]
    #     if not inplace:
    #         ccwht = torch.clone(ccwht)
    #     ccwht[:, :4] += torch.stack([x, y, w, h], dim = -1) * self.pic_size
    #     ccwht[:, :2].clip_(min = 0, max = self.pic_size)
    #     if self.v_rotate_distribution is not None and ccwht.shape[-1] >= 5:
    #         ccwht[:, 4] += (self.v_rotate_distribution.sample(torch.Size([bs])) * (math.pi)  * (torch.randint(0,2,[bs])  * 2 - 1)).to(template)
    #         if ccwht.shape[-1] == 7:
    #             ccwht[:, 5] += (self.v_rotate_distribution.sample(torch.Size([bs])) * (math.pi / 2) * (torch.randint(0,2,[bs])  * 2 - 1)).to(template)
    #             ccwht[:, 6] += (self.v_rotate_distribution.sample(torch.Size([bs])) * (math.pi / 2) * (torch.randint(0,2,[bs])  * 2 - 1)).to(template)
    #     return ccwht


class TransformationData:
    def __init__(self, ccwh_cam, ccwhrrr_patch, color_cam, color_patch):
        self.ccwh_cam = ccwh_cam
        self.ccwhrrr_patch = ccwhrrr_patch
        self.color_cam = color_cam
        self.color_patch = color_patch







    # def __call__(self, fn_attack, background_image):
    #     if self.ccwh_cam is not None:
    #         background_image =
