import math
import numbers
from typing import Optional

import torch
import torchvision
from clip.model import ModifiedResNet, CLIP, VisionTransformer
from timm.layers import Swish
from torch import nn
from torchvision.models import resnet18
from torchvision.ops import Conv2dNormActivation

from dynamic_example.models import reduction, GaussianREG
from dynamic_example.models.detectors import DetectionCaller
from dynamic_example.models.extract_encoding import HookEmb, PositionalEmbedding2d
from dynamic_example.utils import freeze_bn


# from pixel_generator.ldm.models.autoencoder import VQModel


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False, norm=torch.nn.BatchNorm2d, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.a1 = nn.Sequential(Conv2dNormActivation(in_ch, out_ch, norm_layer=norm),
                                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                norm(out_ch) if norm is not None else nn.Identity())
        if residual:
            self.a2 = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = act
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return self.act(self.a1(x) + self.a2(x))
        else:
            return self.act(self.a1(x))


class ResidualBlockDDPM(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_channels=None,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.LazyLinear(out_channels) if time_emb_channels is None else nn.Linear(time_emb_channels,
                                                                                                out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        if t is None:
            t = torch.randn(x.shape[0], self.time_emb.in_features, device=x.device, dtype=x.dtype).clip(min=-4, max=4)
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, residual=False, conv1=False, norm=None):
        super().__init__()

        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        if conv1:
            self.conv = Conv2dNormActivation(in_ch, out_ch, norm_layer=norm)
        else:
            self.conv = ConvBlock(in_ch, out_ch, residual, norm=norm)

    def forward(self, x):
        return self.conv(self.up(x))


class UpTime(nn.Module):
    def __init__(self, in_ch, out_ch, t_ch):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = ResidualBlockDDPM(in_ch, out_ch, time_emb_channels=t_ch, dropout=0.0)

    def forward(self, x, t=None):
        return self.conv(self.up(x), t)


class ResVAE2(nn.Module):
    def __init__(self, in_channels=3, num_filters=[64, 128, 256, 512], latent_dim=256,
                 size=256, flatten=False, residual_dec = True):
        super().__init__()
        assert size % 8 == 0
        resnet = resnet18(pretrained=False, progress=True)
        last_fea_size = size // 32
        if not flatten:
            self.encoder = nn.Sequential(*(list(resnet.children())[:-2]))

        else:
            self.encoder = nn.Sequential(*(list(resnet.children())[:-2] + [
                torch.nn.LazyConv2d(latent_dim // 2, stride=2, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(latent_dim // 2),
                torch.nn.AdaptiveAvgPool2d(2),
                torch.nn.LeakyReLU(),
                torch.nn.Flatten(),
                torch.nn.LazyLinear(latent_dim * 2)]))

        map_size = num_filters[-1]
        self.mu = torch.nn.LazyConv2d(latent_dim, 1) if not flatten else torch.nn.LazyLinear(latent_dim)
        self.mu2 = torch.nn.LazyConv2d(latent_dim, 1) if not flatten else torch.nn.LazyLinear(latent_dim)
        self.log_var = torch.nn.LazyConv2d(latent_dim, 1) if not flatten else torch.nn.LazyLinear(latent_dim)
        self.dec1 = torch.nn.LazyConv2d(map_size, 1) if not flatten else \
            torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim * 4),
                torch.nn.Unflatten(1, (latent_dim // 4, 4, 4)),
                torch.nn.ConvTranspose2d(latent_dim // 4, map_size, stride=2, kernel_size=4, padding=1),
                torch.nn.InstanceNorm2d(map_size),
                torch.nn.LeakyReLU())
        self.dec3 = torch.nn.Sequential(Up(num_filters[3], num_filters[2], residual=residual_dec, norm=nn.InstanceNorm2d),
                                        Up(num_filters[2], num_filters[1], residual=residual_dec, norm=nn.InstanceNorm2d),
                                        Up(num_filters[1], num_filters[0], residual=residual_dec, norm=nn.InstanceNorm2d),
                                        nn.ConvTranspose2d(num_filters[0], num_filters[0] // 2, 4, stride=2, padding=1),
                                        nn.InstanceNorm2d(num_filters[0] // 2),
                                        nn.ReLU(inplace=True),
                                        nn.ConvTranspose2d(num_filters[0] // 2, in_channels, 4, stride=2, padding=1))
        # self.ratio = size * size / latent_dim
        self.latent_shape = size // 32 if not flatten else 1.0  # corresponds to emb_size
        self.ratio = latent_dim * self.latent_shape ** 2 / size / size
        self.flatten = flatten
        self.latent_dim = latent_dim

    def forward(self, x, t=None):
        mu, log_var = self.enc(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.dec(z, t)
        kl_loss = -0.5 * torch.mean(
            1 + log_var.float() - mu.float().pow(2) - log_var.float().clip(min=-40, max=20).exp(), dim=-1)
        kl_loss = kl_loss * self.ratio
        return recon_x, reduction(kl_loss)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp((log_var.float() / 2).clip(min=-20, max=10))
        eps = torch.randn_like(std)
        return mu + (eps * std).clamp(min=-1e4, max=1e4).to(mu.dtype)

    def dec(self, z, t=None):
        # if self.flatten:
        #     z = self.z0.unsqueeze(0) + z
        z = z.clamp(min=-1e4, max=1e4)
        z = self.dec1(z)
        # z = self.dec2(z)
        p = self.dec3(z) + 0.5
        return p.clip(min=0.00001, max=0.99999)

    def enc(self, x, use_mu2=False):
        enc = self.encoder(x)
        mu = self.mu(enc) if not use_mu2 else self.mu2(enc)  # / self.mu.in_features
        log_var = self.log_var(enc)  # / self.log_var.in_features
        return mu, log_var


class ResVAETime(ResVAE2):
    def __init__(self, in_channels=3, num_filters=[64, 128, 256, 512], latent_dim=256, time_dim=2048,
                 size=256, flatten=False):
        super().__init__(in_channels, num_filters, latent_dim=latent_dim, size=size, flatten=flatten)
        time_dim = latent_dim

        self.dec3 = torch.nn.ModuleList([UpTime(num_filters[3], num_filters[2], time_dim),
                                         UpTime(num_filters[2], num_filters[1], time_dim),
                                         UpTime(num_filters[1], num_filters[0], time_dim),
                                         nn.ConvTranspose2d(num_filters[0], num_filters[0] // 2, 4, stride=2,
                                                            padding=1),
                                         nn.InstanceNorm2d(num_filters[0] // 2),
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose2d(num_filters[0] // 2, in_channels, 4, stride=2, padding=1)])

    def dec(self, z, t=None):
        if self.flatten:
            t = z
        z = self.dec1(z)
        # z = self.dec2(z)
        for block in self.dec3:
            z = block(z, t) if isinstance(block, UpTime) else block(z)
        p = z + 0.5
        return p.clip(min=0.00001, max=0.99999)


class VQEncoderDecoder(ResVAE2):
    def __init__(self, in_channels=3, num_filters=[64, 128, 256, 512], latent_dim=256,
                 size=256, flatten=True):
        super().__init__(in_channels, num_filters, latent_dim,
                         size, flatten)
        # self.dec2 = partial(torch.reshape, shape = (-1, map_size, 1, 1))
        self.dec2 = torch.nn.Sequential(Up(num_filters[3], num_filters[1], residual=True),
                                        Up(num_filters[1], 4, residual=True))

        # ldm: LDMTextToImagePipeline = DiffusionPipeline.from_pretrained(
        #     "/home/hujin/.cache/huggingface/hub/models--CompVis--ldm-text2im-large-256/snapshots/30de525ca11a880baea4962827fb6cb0bb268955")
        #
        # self.vqvae: AutoencoderKL = ldm.vqvae
        #
        # for p in self.vqvae.parameters():
        #     p.requires_grad = False
        #
        # freeze_bn(self.vqvae)
        # self.vqvae.eval()
        # self.vqvae.train = lambda x: x

        from diffusers import DiffusionPipeline, AutoencoderTiny


        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        self.vqvae: AutoencoderTiny = vae

        # for p in self.vqvae.parameters():
        #     p.requires_grad = False

        # freeze_bn(self.vqvae)
        # self.vqvae.eval()
        # self.vqvae.train = lambda x: x

    def dec(self, z, t=None):
        z = z.clamp(min=-1e4, max=1e4)
        z = self.dec1(z)
        z = self.dec2(z)
        # z = 1 / self.vqvae.config.scaling_factor * z
        img = self.vqvae.decode(z).sample
        img = (img / 2 + 0.5)

        return img.clip(min=0.00001, max=0.99999)




class TargetTokenizer(nn.Module):
    def __init__(self, emb_dim=256, num_cls=80):
        super().__init__()
        self.emb = torch.nn.Parameter(torch.zeros(num_cls, emb_dim))
        init_(self.emb.data)

    def forward(self, idx):
        return self.emb[idx]


proc_clip_norm = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                  (0.26862954, 0.26130258, 0.27577711))


def proc_clip(clipmodel: CLIP, x, x_extra=None):
    if x_extra is None:
        return clipmodel.encode_image(proc_clip_norm(x))
    return clipmodel.visual(torch.cat([proc_clip_norm(x), x_extra], dim=1).type(clipmodel.visual.conv1.conv1.weight.dtype))


def modify_clip(clip_model, extra_template):
    visual: Optional[ModifiedResNet | VisionTransformer] = clip_model.visual

    conv1_old: torch.nn.Conv2d = visual.conv1
    assert len(extra_template.shape) == 4, "only support BCHW image mask"
    # conv_insert = torch.nn.Conv2d

    class NewConv(nn.Conv2d):
        def __init__(self, c1, **kwargs):
            super().__init__(**kwargs)
            self.conv1 = c1
        def forward(self, x):
            return self.conv1(x[:, :3]) + super().forward(x[:, 3:])
    visual.conv1 = NewConv(conv1_old, in_channels=extra_template.shape[1], out_channels=conv1_old.out_channels,
                                  kernel_size=conv1_old.kernel_size, stride=conv1_old.stride, padding=conv1_old.padding,
                                  bias=True).to(conv1_old.weight)

class EnvEncoder2(nn.Module):
    def __init__(self, extra_channel=1, emb_dim=256, latent_dim=256, create_dec=False, resize=256, fea_dim=2048,
                 introduce_context=True, pos_emb=True, flatten=False, resnet_env="r50_torch", freeze_backbone=False,rand_zadv = False, **kwargs):

        super().__init__()
        # self.resnet_backbone = extract_spark()
        self.input_img_resize = torchvision.transforms.Resize(size=resize, antialias=True)
        self.extra_channel_conv = torch.nn.Conv2d(extra_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                  bias=False)
        self.rand_zadv = rand_zadv
        if resnet_env == "r50_torch":
            self.resnet_backbone = torchvision.models.resnet50(pretrained=True)
        elif resnet_env.__contains__("clip_openai"):
            assert resize == 224
            import clip
            map = {"r50_clip_openai": "RN50", "vitb16_clip_openai": "ViT-B/16"}
            assert resnet_env in map.keys()

            self.resnet_backbone = clip.load(map[resnet_env], jit=False, device = self.extra_channel_conv.weight.device)[0]
            self.clip_modifid = False

        if freeze_backbone:
            for p in self.resnet_backbone.parameters():
                if resnet_env.__contains__("clip_openai"):
                    if p.data_ptr() == self.resnet_backbone.visual.conv1.weight.data_ptr():
                        # exit()
                        continue
                p.requires_grad = False
            freeze_bn(self.resnet_backbone)
        self.backbone_version = resnet_env


        if flatten:
            self.fc1 = nn.Sequential(
                nn.LazyLinear(fea_dim),
                # nn.Dropout(p=0.2, inplace=True),
                Swish(inplace=True))
            self.fc2 = nn.Sequential(
                nn.LazyLinear(fea_dim // 2),
                Swish(inplace=True),
                nn.LazyLinear(fea_dim),
                # nn.Dropout(p=0.1, inplace=True),
                Swish(inplace=True))
            self.fc_emb = nn.Sequential(
                Swish(inplace=True),
                nn.LazyLinear(fea_dim),
                # nn.Dropout(p=0.2, inplace=True),
                # nn.Dropout(p=0.1, inplace=True),
                Swish(inplace=True))
            self.fc3 = nn.Sequential(
                nn.LazyLinear(fea_dim),
                # nn.Dropout(p=0.1, inplace=True),
                Swish(inplace=True),
                nn.LazyLinear(latent_dim * 2 if rand_zadv else latent_dim))
            
            

        else:
            self.fc1 = nn.Sequential(
                nn.LazyLinear(fea_dim),
                # nn.Dropout(p=0.1, inplace=True),
                Swish(inplace=True),
                nn.Linear(fea_dim, fea_dim))

            self.fc2 = nn.Sequential(
                nn.LazyLinear(fea_dim // 2),
                # nn.Dropout(p=0.1, inplace=True),
                Swish(inplace=True),
                nn.Linear(fea_dim // 2, fea_dim))
            
            
            self.fc3 = nn.Sequential(
                nn.LazyLinear(fea_dim),
                # nn.Dropout(p=0.1, inplace=True),
                Swish(inplace=True),
                nn.LazyLinear(latent_dim))

            dim_pos_emb = 64
            self.pos_embeding = PositionalEmbedding2d(dim_pos_emb) if pos_emb else None
            self.pre = ResidualBlockDDPM(latent_dim, emb_dim)
            self.atten = AttentionBlock(emb_dim) if not introduce_context else SpatialTransformer(
                emb_dim + dim_pos_emb if pos_emb else emb_dim, emb_dim, n_heads=emb_dim // 32, d_head=32, context_dim=emb_dim)
            self.post = ResidualBlockDDPM(emb_dim, latent_dim)
            
            
            
            
        self.mu = torch.nn.Linear(fea_dim, fea_dim)

        self.flatten = flatten

        if create_dec:
            # out_mid_channel = [emb_dim * 2, emb_dim, emb_dim / 2]
            # out_channel = 3
            # self.dec1 = nn.Sequential(
            #     nn.ConvTranspose2d(latent_dim, out_mid_channel[0], kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm2d(out_mid_channel[0]),
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(out_mid_channel[0], out_mid_channel[1], kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm2d(out_mid_channel[1]),
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(out_mid_channel[1], out_mid_channel[2], 4, stride=2, padding=1),
            #     nn.BatchNorm2d(out_mid_channel[2]),
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(out_mid_channel[2], out_channel, 4, stride=2, padding=1),
            #     nn.Sigmoid()
            # )
            self.dec_motion = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.Dropout(inplace=True, p = 0.1),
                nn.ReLU(True),
                nn.Linear(latent_dim * 2, latent_dim * 2),
                nn.ReLU(True),
                nn.Linear(latent_dim * 2, 6),  # return wh at each point
            )

        self.likehood_gaussian = GaussianREG()

        self.ratio = latent_dim * (resize // 32) ** 2 / resize / resize
        self.emb_bn = nn.LayerNorm(fea_dim)
        self.fea_bn = nn.LayerNorm(fea_dim)
        self.detector = None
        self.fea_dim = fea_dim

    def lookup(self, x, x0, ratio):
        xw = x.reshape(x.shape[0], -1)
        xw = self.codebook(xw)
        return x0 * (1 - ratio).reshape(-1, 1, 1, 1) + xw.reshape(*x0.shape) * ratio.reshape(-1, 1, 1, 1)

    def forward_backbone(self, x0, x_extra, mask=None, task_ratio=None):
        x0 = self.input_img_resize(x0)

        if self.backbone_version == "r50_torch":
            x = self.resnet_backbone.conv1(x0)
            if isinstance(x_extra, torch.Tensor):
                x_extra = self.input_img_resize(x_extra)
                x1 = self.extra_channel_conv(x_extra)
                x = x + x1
            elif isinstance(x_extra, numbers.Number):  #
                x = x + x_extra

            x = self.resnet_backbone.bn1(x)
            # x = self.resnet_backbone.act1(x)
            x = self.resnet_backbone.relu(x)
            x = self.resnet_backbone.maxpool(x)

            x1 = self.resnet_backbone.layer1(x)  # .detach()
            x2 = self.resnet_backbone.layer2(x1)  # .detach()
            x3 = self.resnet_backbone.layer3(x2)  # .detach()
            x4 = self.resnet_backbone.layer4(x3)  # .detach()
            x = self.resnet_backbone.avgpool(x4)
            x = torch.flatten(x, 1)
        elif isinstance(self.resnet_backbone, CLIP):
            if x_extra is not None:
                if not self.clip_modifid:
                    modify_clip(self.resnet_backbone, x_extra)
                    self.clip_modifid = True
                x_extra = self.input_img_resize(x_extra)
                x = proc_clip(self.resnet_backbone, x0, x_extra).to(x0)
            else:
                x = proc_clip(self.resnet_backbone, x0).to(x0)
        else:
            raise NotImplementedError()
        x = self.fc1(x)
        if mask is not None:
            ts_emb0 = - torch.log((1 - task_ratio).clip(min=math.exp(-10)))
            ts_emb1 = - torch.log(task_ratio.clip(min=math.exp(-10)))
            x = x + self.fc2(torch.cat([mask.to(x) / mask.shape[-1] * 5, ts_emb0, ts_emb1], dim=-1))
        elif task_ratio is not None:
            ts_emb0 = - torch.log((1 - task_ratio).clip(min=math.exp(-10)))
            ts_emb1 = - torch.log(task_ratio.clip(min=math.exp(-10)))
            x = x + self.fc2(torch.cat([ts_emb0, ts_emb1], dim=-1))
        # if task_ratio is not None:
        #     x = x * task_ratio
        # x = torch.nn.GELU()(x)
        return x

    def attach_vae(self, dec: ResVAE2):
        self.enc_vae = dec.enc
        self.dec_vae = dec.dec

    def attach_detector(self, detector: DetectionCaller, emb: HookEmb):
        self.detector = detector
        self.emb_hook = emb

    def forward(self, img: Optional[torch.Tensor], task_ratio, patch, img_extra_channel=None, emb=None,
                return_pos=False, return_atten=False, return_emb = False,
                use_vae=True, detach_vae=False, mask=None, rand_vae=False):
        task_ratio = task_ratio[:, 0, None]


        mu_p, logvar_p = self.enc_vae(patch, use_mu2=True)
        emb = mu_p

        if self.flatten:
            emb, regularization = self.extract_emb_flatten(emb, img, img_extra_channel, mask, task_ratio)
            time_emb = emb
        else:
            emb, time_emb,  regularization = self.extract_emb_fea(emb, img, img_extra_channel, mask, task_ratio)

        if self.dec_vae is not None and use_vae:

            emb = torch.tanh(emb / 2) * 2  # limit emb into ±2σ
            # emb = self.emb_bn(emb.unsqueeze(-1))[:, :, 0] * 0.1
            # if not self.flatten:
            #     fea = emb# torch.tanh(self.mu(fea))
            # else:
            #     fea = emb
            # print(time_emb.shape)
            ret = [self.dec_vae(emb, t=time_emb), regularization, 0.0]

        else:
            patch = self.dec1(emb)
            ret = [patch, emb]
        if return_pos:
            motion = self.dec_motion(emb.float())
            ret.append(motion)
        if return_atten:
            ret.append(None)
        if return_emb:
            ret.append(emb)
        return tuple(ret)

    def extract_emb_fea(self, emb, img, img_extra_channel, mask, task_ratio):
        if img is not None:
            fea = self.forward_backbone(img, img_extra_channel, mask=mask, task_ratio=task_ratio)
        else:
            fea = torch.zeros(emb.shape[0], self.fea_dim).to(emb)
            
        # if self.training:
        #     rnd = torch.rand_like(emb[:, :1, :])
        #     emb = torch.where(rnd < 0.3, torch.zeros_like(emb) , emb)
            # emb[] = 0
            
        emb0 = fea
        emb = self.pre(emb, fea)
        context = fea.reshape(emb.shape[0], -1, emb.shape[1])
        if self.pos_embeding is not None:
            emb = self.pos_embeding(emb)# , dim=1)
        # if self.detector is not None:
        #     self.emb_hook.clean()
        #     self.detector(img)
        #     context = torch.cat([context, self.emb_hook()], dim=1)
        #     self.emb_hook.clean()
        emb = self.atten(emb, context)
        emb = self.post(emb, fea)
        emb_t = self.fc3(emb0)
        return emb, emb_t, self.likehood_gaussian(emb) + self.likehood_gaussian(emb_t)

    def extract_emb_flatten(self, emb, img, img_extra_channel, mask, task_ratio):
        if img is not None:
            fea = self.forward_backbone(img, img_extra_channel, mask=mask, task_ratio=task_ratio)
            emb = self.fc_emb(emb)
            emb = self.emb_bn(emb)
            fea = self.fea_bn(fea)
            fea0 = torch.zeros_like(fea)
            emb = self.fc3(torch.cat([torch.where(task_ratio > 0, fea, fea0), emb], dim=1))
        else:
            emb = self.fc_emb(emb)
            emb = self.emb_bn(emb)
            fea = torch.zeros_like(emb)
            emb = self.fc3(torch.cat([fea, emb], dim=1))

        if self.rand_zadv:
            mu, logvar = emb[..., :emb.shape[-1] // 2], emb[..., emb.shape[-1] // 2 : ]
            emb = ResVAE2.reparameterize(mu, logvar)
            # if not self.training: # TODO
            #     emb = mu

            kl_loss = -0.5 * torch.mean(
                1 + logvar.float() - mu.float().pow(2) - logvar.float().clip(min=-40, max=20).exp(), dim=-1)
            return emb, kl_loss

        kl_loss = self.likehood_gaussian(emb)
        return emb, kl_loss


from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, out_channels, n_heads=8, d_head=32,
                 depth=2, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels <= in_channels  # residual: channel clip
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
             for _ in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in[:, :self.out_channels]


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.LazyLinear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res
