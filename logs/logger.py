import torch 
import wandb

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, file_name, enable_wandb=True):
        self.path = f'logs/logfiles/{file_name}'
        self.writer = SummaryWriter(self.path,filename_suffix=file_name)
        self.enable_WandB = enable_wandb
    
    def add_image(self, tag, image, step):
        grid_im = utils.make_grid(image, normalize=True)
        self.writer.add_image(tag=tag, img_tensor= grid_im, global_step=step) 
        if self.enable_WandB :
            d = {tag: wandb.Image(grid_im.cpu().data)}
            wandb.log(d)

    def add_scalar(self, tag, scalar, step):
        self.writer.add_scalar(tag=tag,scalar_value=scalar, global_step=step)
        if self.enable_WandB :
            d = {tag: scalar}
            wandb.log(d)
            
    def add_list_images(self, tag, list_of_images, step):
        for name, im in list_of_images.items():
            self.add_image(f'{tag}_{name}', im, step)
    
    def close(self):
        self.writer.close()