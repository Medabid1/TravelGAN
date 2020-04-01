
import torch 


a = torch.Tensor(size=(1, 1, 32, 32))
conv = torch.nn.Conv2d(1, 1, 3, 2, padding=1)
tconv = torch.nn.ConvTranspose2d(1, 1, 4, 2, padding=1)

print('init', a.size())

for _ in range(4):
    out = conv(a)
    a = out
    print('conv', a.size())

for _ in range(4):
    out = tconv(a)
    a = out
    print('T', a.size())