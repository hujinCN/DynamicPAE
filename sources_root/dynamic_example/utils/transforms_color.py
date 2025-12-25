from typing import List

import torch
import torchvision.transforms._functional_tensor
import torchvision.transforms.functional as F
from torch import Tensor


def adjust_brightness(img: Tensor, brightness_factor: Tensor) -> Tensor:
    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: Tensor) -> Tensor:
    c = F.get_dimensions(img)[0]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    if c == 3:
        mean = torch.mean(F.rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    else:
        mean = torch.mean(img.to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_saturation(img: Tensor, saturation_factor: Tensor) -> Tensor:
    if F.get_dimensions(img)[0] == 1:  # Match PIL behaviour
        return img

    return _blend(img, F.rgb_to_grayscale(img), saturation_factor)


def adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input img should be a Tensor.")

    # if gamma < 0:
    #     raise ValueError("Gamma should be a non-negative real number")

    result = img
    dtype = img.dtype
    if not torch.is_floating_point(img):
        result = F.convert_image_dtype(result, torch.float32)

    result = (gain * result ** gamma.reshape(-1,1,1,1)).clamp(0, 1)

    result = F.convert_image_dtype(result, dtype)
    return result


def _get_gaussian_kernel1d(kernel_size: int, sigma: torch.Tensor) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5
    sigma = sigma.float()
    sigma = sigma.clip(min  = 0.001)
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma)
    x = x.expand(sigma.shape[0], x.shape[0])
    sigma = sigma.unsqueeze(-1)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum(dim = -1, keepdim = True)
    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma).to(device, dtype=dtype)
    kernel2d = torch.matmul(kernel1d_y[:, :, None], kernel1d_x[:, None, :])
    return kernel2d


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Tensor) -> Tensor:

    bs = img.shape[0]
    if bs == 0:
        return img
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.reshape(bs, 1, kernel.shape[-2], kernel.shape[-1])

    # img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img:Tensor = torch.nn.functional.pad(img, padding, mode="reflect")
    img = img.permute(1, 0, 2, 3) # c b h w
    img = torch.nn.functional.conv2d(img, kernel, groups=bs)
    img = img.permute(1, 0, 2, 3) # b c h w
    return img
def _blend(img1: Tensor, img2: Tensor, ratio: Tensor) -> Tensor:
    # ratio = float(ratio)
    ratio = ratio.reshape(-1, 1, 1,1)
    bound = torchvision.transforms._functional_tensor._max_value(img1.dtype)
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

class ColorTrans(torch.nn.Module):
    def __init__(self, kernel_size_gaussian = 3):
        super().__init__()
        self.kernel_sz = [kernel_size_gaussian, kernel_size_gaussian]

    def forward(self, imgs, bcsg_gauss_noise):
        b, c, s, g, gau, n = [i.squeeze(-1) for i in torch.split(bcsg_gauss_noise, 1, dim = -1)]
        # imgs = adjust_gamma(imgs, g)
        imgs = adjust_saturation(imgs, s)
        imgs = adjust_contrast(imgs, c)
        imgs = adjust_brightness(imgs, b)
        imgs = imgs + n.reshape(-1, 1, 1, 1) * torch.randn_like(imgs)
        mask = gau > 1e-4
        imgs[mask] = gaussian_blur(imgs[mask].float(), kernel_size=self.kernel_sz, sigma=gau[mask]).to(imgs)
        imgs = torch.clamp(imgs, 0.0, 1.0)
        return imgs

