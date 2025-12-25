import warnings
from functools import partial

import torch
import torch.nn as nn

# See  https://discuss.pytorch.org/t/modify-module-gradients-during-backward-pass/158696

def clip_grad(tensor:torch.Tensor, clip_rate = 1e-3):
    if tensor.requires_grad:
        tensor.register_hook(lambda grad : torch.clamp(grad,min=-clip_rate,max=clip_rate))
def norm_grad(tensor:torch.Tensor, clip_rate = 1e-3):
    if tensor.requires_grad:
        tensor.register_hook(lambda grad : torch.norm(grad,dim = 1, keepdim=True, p = 2) * clip_rate)
def sgn_grad(tensor, eps = 1e-3):
    if tensor.requires_grad:
        tensor.register_hook(lambda grad : torch.sign(grad) * eps)



class GradRescale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = None

    def get_scale(self):
        """
        @return: torch.Tensor (Batch size,)
        """
        return self.scale

    def set_scale(self, scale: torch.Tensor):
        """
        @param scale: torch.Tensor (Batch size,)
        """
        self.scale = scale

    def forward(self, x: torch.Tensor):
        def get_grad(x0):
            len_dim = len(x0.shape)
            dim = list(range(1, len_dim))
            x_len = torch.mean(x0**2, dim = dim, keepdim = True) ** 0.5
            shape_factor = [-1] + [1] * (len_dim - 1)
            scale = self.get_scale()
            scale = scale.reshape(*shape_factor)
            #
            # # blackbox test
            # x = torch.rand_like(x)


            x = x0 / (x_len + 1e-10)
            x = x.clip(min  = -10, max = 10)
            x_len = torch.sum(x**2, dim = dim, keepdim = True) ** 0.5
            x = x / (x_len + 1e-10) # ||x||_2 = 1.0
            xs = (x * scale).to(x0)
            has_nan = torch.isnan(xs).any()
            has_inf = torch.isinf(xs).any()
            if has_inf or has_nan:
                warnings.warn(f"NAN detected: {has_nan}, INF detected: {has_inf}")
                xs = torch.nan_to_num(xs, nan = 0, posinf=0, neginf=0)
            # print(xs.mean(dim = (1,2,3)))
            return xs

        x = torch.clone(x)
        if x.requires_grad:
            x.register_hook(get_grad)
        return x


class GradNorm(torch.nn.Module):
    """
    GradNorm：Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    """
    def __init__(self, num_stages = 1):
        super().__init__()
        self.norms = torch.nn.ModuleList([torch.nn.BatchNorm2d(1) for _ in range(num_stages)])
        for n in self.norms:
            n.running_mean.fill_(1)
        self.cur_stage = 0
        self.num_stages = num_stages
        self.buf = None
    def get_stage(self):
        return self.cur_stage
    def set_optim_stage(self, i):
        self.cur_stage = i

    def forward(self, img: torch.Tensor, loss_ratio: torch.Tensor):
        self.buf = loss_ratio
        img_grad = torch.clone(img)
        shape_factor = [-1] + [1] * (len(img.shape) - 1)
        def norm_l2(grad):
            l2_norm = torch.sqrt(self.norms[self.get_stage()].running_mean).unsqueeze(-1).unsqueeze(-1)
            num_grad = 1
            for x in grad.shape:
                num_grad *= x
            self.norms[self.get_stage()]((grad * num_grad)**2)
            return (grad / l2_norm * self.buf[:, self.get_stage()].reshape(*shape_factor)).to(grad)
        def rescale(grad):
            # l2_norm = torch.sqrt(self.norms[self.get_stage()].running_mean).unsqueeze(-1).unsqueeze(-1)
            # self.norms[self.get_stage()](grad**2)
            return (grad * self.buf[:, self.get_stage()].reshape(*shape_factor)).to(grad)

        if img.requires_grad:
            img_grad.register_hook(norm_l2)
        return img_grad

# usage: base_model.apply(partial(set_loss, loss=***, stage=***)
def set_loss(model: nn.Module, loss_scale, stage):
    if isinstance(model, GradNorm):
        model.set_optim_stage(stage)
    if loss_scale is not None and isinstance(model, GradRescale):
        model.set_scale(loss_scale)

# def skewness(x, x_mean):
#
#     return x - x_mean / x - x_mean

class skip_grad:
    def __init__(self, model: nn.Module) -> None:
        self.grad_state = []
        self.model = model

    def __enter__(self) -> None:
        for p in self.model.parameters():
            self.grad_state.append(p.requires_grad)
            p.requires_grad = False


    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for i, p in enumerate(self.model.parameters()):
            p.requires_grad = self.grad_state[i]




if __name__ == '__main__':
    gn = GradNorm(num_stages=2)
    gr = GradRescale()
    loss_ratio = torch.tensor([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])

    # test grad norm
    for _ in range(100):
        img_in0 = torch.rand(3, 3, 10, 10) * 0.1 + torch.ones(3, 3, 10, 10) * 0.9
        # img_in0[1] += 2
        img_in0.requires_grad = True
        img_in = gn(img_in0, loss_ratio)

        loss_1 = (img_in ** 2).sum() * 10
        gn.apply(partial(set_loss, loss_scale = loss_1, stage = 0))
        loss_1.backward()

        print()
        print(img_in0.grad.data.sum(dim = (1, 2, 3)))
        img_in0.grad.data *= 0

        loss_2 = (img_in ** 3).sum()
        gn.apply(partial(set_loss, loss_scale = loss_2, stage = 1))
        loss_2.backward()

        print(img_in0.grad.data.sum(dim = (1, 2, 3)))
        img_in0.grad.data *= 0
    print("====")
    # test grad rescale: loss: small diff, but grad largely influenced by uncontrollable input
    img_in0 = torch.rand(3, 3, 10, 10) * 0.1 + torch.ones(3, 3, 10, 10) * 0.9
    img_in0[1] += 1
    img_in0.requires_grad = True
    img_in1_noise = torch.ones(3, 3, 10, 10) * 10
    img_in1_noise[2] = .1
    img_in = img_in0 * img_in1_noise + (-1) * img_in1_noise + 10
    loss_1 = torch.log((img_in).sum(dim = (1,2,3))) * 10
    loss_1.sum().backward()
    print(img_in0.grad.data.sum(dim=(1, 2, 3)))
    img_in0.grad.data *= 0

    img_in0 = torch.rand(3, 3, 10, 10) * 0.1 + torch.ones(3, 3, 10, 10) * 1.0
    img_in0[1] += 1
    img_in0.requires_grad = True
    img_in = gr(img_in0)
    img_in1_noise = torch.ones(3, 3, 10, 10) * 10
    img_in1_noise[2] = .1
    img_in = img_in * img_in1_noise + (-1) * img_in1_noise + 10
    loss_1 = torch.log(((img_in)).sum(dim = (1,2,3))) * 10
    gr.apply(partial(set_loss, loss_scale=1 / ((img_in)).sum(dim = (1,2,3)), stage=1))
    loss_1.sum().backward()

    print(img_in0.grad.data.sum(dim=(1, 2, 3)))
    img_in0.grad.data *= 0


