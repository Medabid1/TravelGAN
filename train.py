
import torch
import torch.nn.functional as F 

from itertools import combinations
import numpy as np 

pairs = np.asarray(list(combinations(list(range(10)), 2)))
print(pairs)
print(pairs[:,0] )
print(pairs[:, 1])