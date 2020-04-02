from configparser import ConfigParser



class parser:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.read(path)

    def to_dict(self):
        c = {}
        opt = self.config['OPT']
        c['gen_lr'] = opt.getfloat('gen_lr')
        c['dis_lr'] = opt.getfloat('dis_lr')
        c['gbeta1'] = opt.getfloat('gbeta1')
        c['gbeta2'] = opt.getfloat('gbeta2')
        c['dbeta1'] = opt.getfloat('dbeta1')
        c['dbeta2'] = opt.getfloat('dbeta2')
        c['batch_size'] = opt.getint('batch_size')
        c['epochs'] = opt.getint('epochs')
        c['step_size'] = opt.getint('step_size')
        
        model = self.config['MODEL']
        c['iter_sample'] = model.getint('iter_sample')
        c['iter_log'] = model.getint('iter_log')
        c['device'] = model['device']
        c['enable_wandb'] = model.getboolean('enable_wandb')
        c['logfile'] = model['logfile']
        c['checkpoint_path'] = model['checkpoint_path']
        c['checkpoint_iter'] = model.getint['checkpoint_iter']
        c['image_size'] = model.getint('image_size')
        c['in_channels'] = model.getint('in_channels')
        c['num_feat'] = model.getint('num_feat')
        c['num_repeat'] = model.getint('num_repeat')
        c['num_res'] = model.getint('num_res')

        loss = self.config['LOSS']
        c['gen_adv_loss_w'] = loss.getint('gen_adv_loss_w')
        c['siamese_loss_w'] = loss.getint('siamese_loss_w')
        c['gamma'] = loss.getfloat('gamma')

        return c