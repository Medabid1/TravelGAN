import torch.nn.init as init
from torchvision import ImageFolder 

def get_DataLoader_fromFolder(path, batch_size, transform=None):
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader
    
def get_indices(dataset,class_name):
    indices =  []
    try : 
        labels = dataset.labels
    except :
        labels = dataset.targets
    for i in range(len(labels)):
        if labels[i] == class_name:
            indices.append(i)
    return indices

def cifar10_class_loader(class1, class2, transform, batch_size):
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
    return two dataloader, class1 and class2 dataloaders.
    """
    assert class1 =! class2, 'the two classes should be different'

    data = CIFAR10('data/', download=True, transform=transform)
    class1_idx = get_indices(data, class1)
    class2_idx = get_indices(data, class2)
    class1_loader = Data.DataLoader(data, batch_size=batch_size, sampler = Data.sampler.SubsetRandomSampler(bird_idx))
    class2_loader = Data.DataLoader(data, batch_size=batch_size, sampler = Data.sampler.SubsetRandomSampler(plane_idx))

    return class1_loader, class2_loader

def weights_init(init_type='kaiming'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun