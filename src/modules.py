import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from utils import weights_init
from collections import OrderedDict
from itertools import combinations

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
        self.is_attention = attention
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
        if self.is_attention :
            x = x + out * self.attention(out)
        else :
            x = x + out 
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, num_feat, num_res):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(num_feat))
        layers.append(nn.ReLU())

        curr_dim = num_feat
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        
        for _ in range(num_res):
            layers.append(ResidualBlock(curr_dim, curr_dim))


        for _ in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, in_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        self.apply(weights_init())

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, in_channels, num_feat=64, num_repeat=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = num_feat
        for _ in range(1, num_repeat):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)
        self.apply(weights_init())

    def forward(self, x):
        h = self.main(x)
        out = self.conv1(h)
        return out

    def calc_dis_loss(self, x_real, x_fake):
        real_pred = self.forward(x_real)
        fake_pred = self.forward(x_fake)
        loss = torch.mean((real_pred - 1)**2) + torch.mean((fake_pred - 0)**2)
        return loss
    
    def calc_gen_loss(self, x):
        pred = self.forward(x)
        loss = torch.mean((pred - 1)**2)
        return loss 

class SiameseNet(nn.Module):
    def __init__(self, image_size, in_channels, num_feat=64, num_repeat=5, gamma=10):
        super().__init__()
        layers = []
        self.gamma = gamma 
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = num_feat
        for _ in range(1, num_repeat):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        in_feat = image_size // 2**(num_repeat)

        self.main = nn.Sequential(*layers)
        self.linear = nn.Linear(curr_dim*in_feat**2, 1024)
        self.apply(weights_init())

    def _forward(self, x1, x2):
        latent1 = self.main(x1)
        latent2 = self.main(x2)
        latent1 = self.linear(latent1.flatten(1))
        latent2 = self.linear(latent2.flatten(1))
        return latent1, latent2
        
    def calc_loss(self, x1, x2):
        pairs = np.asarray(list(combinations(list(range(x1.size(0))), 2)))
        latent1, latent2 = self._forward(x1, x2)
        v1 = latent1[pairs[:,0]] - latent1[pairs[:,1]]
        v2 = latent2[pairs[:,0]] - latent2[pairs[:,1]]
        distance = F.mse_loss(v1, v2) - torch.mean(F.cosine_similarity(v1, v2)) 
        return distance + self.margin_loss(v1)

    def margin_loss(self, v1):
        return F.relu(torch.mean(self.gamma - torch.norm(v1, dim=1))
