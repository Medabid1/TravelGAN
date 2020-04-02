import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from src.modules import Generator, Discriminator, SiameseNet
from tqdm import tqdm



class TravelGan:
    def __init__(self, config):
        self.config = config 
        self.dis = Discriminator(config['in_channels'], num_feat=config['num_feat'], num_repeat=config['num_repeat'])
        self.gen = Generator(config['in_channels'], num_feat=config['num_feat'], num_res=config['num_res'])
        self.siamese = SiameseNet(config['image_size'], config['in_channels'], num_feat=config['num_feat'], 
                                  num_repeat=config['num_repeat'], gamma=config['gamma'])
        
        gen_params = list(self.gen.parameters()) + list(self.siamese.parameters())
        self.opt_gen = optim.Adam(gen_params, lr=config['gen_lr'], betas=(config['gbeta1'], config['gbeta2']))
        self.opt_dis = optim.Adam(self.dis.parameters(), lr=config['dis_lr'], betas=(config['dbeta1'], config['dbeta2']))

        self.gen_scheduler = optim.lr_scheduler.StepLR(self.opt_gen, config['gen_sch_step_size'], gamma=0.1)
        self.dis_scheduler = optim.lr_scheduler.StepLR(self.opt_dis, config['dis_sch_step_size'], gamma=0.1)

    
    def _train_epoch(self, loaderA, loaderB):
        
        for _, (x_a, x_b) in enumerate(zip(loaderA, loaderB)):
            
            if isinstance(x_a, (tuple, list)):
                x_a = x_a[0]
            if isinstance(x_b, (tuple, list)):
                x_b = x_b[0]

            self.opt_dis.zero_grad()
            x_ab = self.gen(x_a)
            dis_loss = self.dis.calc_dis_loss(x_b, x_ab.detach())
            dis_loss.backward()
            self.opt_dis.step()

            self.opt_gen.zero_grad()
            gen_adv_loss = self.dis.calc_gen_loss(x_ab)
            gen_siamese_loss = self.siamese.calc_loss(x_a, x_ab)

            gen_loss = self.config['gen_adv_loss_w'] * gen_adv_loss + \
                       self.config['siamese_loss_w'] * gen_siamese_loss
            
            gen_loss.backward()
            self.opt_gen.step()
    
    def train(self, loaderA, loaderB):
        for i in tqdm(range(self.config['epochs'])):
            self._train_epoch(loaderA, loaderB)
            self.gen_scheduler.step()
            self.dis_scheduler.step()

            if i % self.config['checkpoint_iter'] :
                self.save()

    def save(self): 
        torch.save({'gen' : self.gen.state_dict(),
                    'dis' : self.dis.state_dict(),
                    'siamese' : self.siamese.state_dict(),
                    'gen_opt' : self.opt_gen.state_dict(),
                    'dis_opt' : self.opt_dis.state_dict(),
                    'gen_scheduler' : self.gen_scheduler.state_dict(),
                    'dis_scheduler' : self.dis_scheduler.state_dict()
        }, self.config['checkpoint_path'])

    def load(self):
        checkpoint = torch.load(self.config['checkpoint_path'])
        self.gen.load_state_dict(checkpoint['gen'])
        self.dis.load_state_dict(checkpoint['dis'])
        self.siamese.load_state_dict(checkpoint['siamese'])
        self.opt_dis.load_state_dict(checkpoint['dis_opt'])
        self.opt_gen.load_state_dict(checkpoint['gen_opt'])
        self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
        self.dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])