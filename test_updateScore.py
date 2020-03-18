import torch
import torch.nn as nn
import numpy as np

def updateScore(mask):
    with torch.no_grad():
        K = 2
        mask_flatten = mask.flatten()
        mask1 = torch.eq(mask_flatten, 1)
        mask2 = torch.ne(mask1, True)
        x = -mask.grad
        x = x.flatten()
        x1 = torch.masked_select(x, mask1)
        x2 = torch.masked_select(x, mask2)
        topk_min, idx1 = torch.topk(x1, K, largest=False)
        topk_max, idx2 = torch.topk(x2, K)
        index_nonzero = torch.nonzero(mask1)
        index_zero = torch.nonzero(mask2)
        for i in range(K):
            if topk_max[i] - topk_min[i] > 0:
                mask_flatten[index_nonzero[idx1[0]]] = False
                mask_flatten[index_zero[idx2[0]]] = True

torch.manual_seed(1)
mask = nn.Parameter(torch.Tensor(2,3))
torch.nn.init.zeros_(mask)
mask.grad = torch.tensor(np.array([[1., 2., 3.], [4., 5., 6.]]), dtype=torch.float32)


with torch.no_grad():
    m = torch.Tensor(2,3).uniform_()
    m = m < 0.5
    mask.masked_fill_(m, 1)
updateScore(mask)
