
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

source_path = 'data/vangogh2photo/real'
target_path = 'data/vangogh2photo/vangogh'

pic_loader = get_DataLoader_fromFolder(source_path, config['batch_size'])
monet_loader = get_DataLoader_fromFolder(target_path, config['batch_size'])

model = TravelGan(config, logger)

model.train(pic_loader, monet_loader)
