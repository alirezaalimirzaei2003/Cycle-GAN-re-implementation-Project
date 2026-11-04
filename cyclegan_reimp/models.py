

import functools
import torch
import torch.nn as nn



# Weight initialization 

def init_weights(module, gain=0.02):
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
        nn.init.normal_(module.weight.data, 0.0, gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname or "BatchNorm2d" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 1.0, gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


def init_net(net, gain=0.02, device="cpu"):
    net = net.to(device)
    net.apply(lambda m: init_weights(m, gain))
    return net



# Residual block: Conv(3x3) with reflection padding

class ResidualBlock(nn.Module):
    def __init__(self, dim, use_dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)  # skip connection



# Generator
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, use_dropout=False):
        """
        n_blocks: 9 for 256x256 (paper default), 6 for 128x128 images.
        """
        super().__init__()
        assert n_blocks >= 0

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        ]

        # Downsampling: d128, d256
        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]

        # Residual blocks: R256 x n_blocks (operating at 1/4 resolution)
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * mult, use_dropout=use_dropout)]

        # Upsampling: u128, u64
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                    padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2), affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]

        # Output layer:
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



# Discriminator: 70x70 PatchGAN  
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1

        # First layer: no normalization 
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        
        #  we use LSGAN (MSE) loss
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



# LSGAN objective 
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        target = (self.real_label if target_is_real else self.fake_label).expand_as(prediction)
        return self.loss(prediction, target)


def define_generator(input_nc=3, output_nc=3, ngf=64, n_blocks=9, device="cpu"):
    return init_net(ResnetGenerator(input_nc, output_nc, ngf, n_blocks), device=device)


def define_discriminator(input_nc=3, ndf=64, n_layers=3, device="cpu"):
    return init_net(NLayerDiscriminator(input_nc, ndf, n_layers), device=device)
