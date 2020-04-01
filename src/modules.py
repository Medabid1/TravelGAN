import torch 
import torch.nn as nn 

from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, in_feat, num_feat, reduction=16, attention=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_feat, num_feat, kernel_size=3, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(num_feat, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(num_feat, in_feat, kernel_size=3, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(in_feat, affine=True, track_running_stats=True))
        self.main = nn.Sequential(*layers)
        if attention :
            attentionlayer = [] 
            attentionlayer.append(nn.AdaptiveAvgPool2d(1))
            attentionlayer.append(nn.Conv2d(in_feat, in_feat//reduction, kernel_size=1))
            attentionlayer.append(nn.ReLU())
            attentionlayer.append(nn.Conv2d(in_feat//reduction, in_feat, kernel_size=1))
            attentionlayer.append(nn.Sigmoid())
            self.attention = nn.Sequential(*attentionlayer)

    def forward(self, x):
        out = self.main(x)
        x = x + out * self.attention(out)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, init_feat, num_res):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, init_feat, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(init_feat, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())

        curr_dim = init_feat
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        
        for _ in range(num_res):
            layers.append(ResidualBlock(curr_dim, curr_dim))


        for _ in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, in_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, num_feat=64, repeat_num=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = num_feat
        for _ in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out = self.conv1(h)
        return out