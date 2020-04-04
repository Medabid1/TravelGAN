
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
from utils import get_DataLoader_fromFolder

parser = parser('config/config.ini')
config = parser.to_dict()
logger = Logger(config['logfile'], config['enable_wandb'])





source_path = '/gel/usr/maabi11/data/data/vangogh2photo/trainB'
target_path = '/gel/usr/maabi11/data/data/vangogh2photo/trainA'

pic_loader = get_DataLoader_fromFolder(source_path)
monet_loader = get_DataLoader_fromFolder(target_path)

model = TravelGan(config, logger)

model.train(pic_loader, monet_loader)
