
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data 

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from utils import get_indices
from config.parser import parser
from src.travelgan import TravelGan
from logs.logger import Logger


parser = parser('config/config.ini')
config = parser.to_dict()
logger = Logger(config['logfile'], config['enable_wandb'])

""" 
    planes = 0
    cars = 1
    bird = 2
    cat = 3										
    deer = 4										
    dog = 5									
    frog = 6										
    horse = 7 										
    ship = 8 										
    truck = 9
"""

#======= 
# Cifar 10 
#data = CIFAR10('data/', download=True, transform=transforms.Compose([transforms.ToTensor(),
#                                                                     transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]))
#bird_idx = get_indices(data, 2)
#plane_idx = get_indices(data, 0)
#bird_loader = Data.DataLoader(data, batch_size=config['batch_size'], sampler = Data.sampler.SubsetRandomSampler(bird_idx))
#plane_loader = Data.DataLoader(data, batch_size=config['batch_size'], sampler = Data.sampler.SubsetRandomSampler(plane_idx))

model = TravelGan(config, logger)

model.train(bird_loader, plane_loader)
