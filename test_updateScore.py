import torch
import torch.nn as nn
import numpy as np

def updateScore(mask):
    with torch.no_grad():
        K = 2
        mask_flatten = mask.flatten()
        print(mask_flatten)
        mask1 = torch.eq(mask_flatten, 1)
        print(mask1)
        mask2 = torch.ne(mask1, True)
        print(mask2)
        x = -mask.grad
        x = x.flatten()
        print(x)
        x1 = torch.masked_select(x, mask1)
        print(x1)
        x2 = torch.masked_select(x, mask2)
        print(x2)
        topk_min, idx1 = torch.topk(x1, K, largest=False)
        print(topk_min, idx1)
        topk_max, idx2 = torch.topk(x2, K)
        print(topk_max, idx2)
        index_nonzero = torch.nonzero(mask1)
        print(index_nonzero)
        index_zero = torch.nonzero(mask2)
        print (index_zero)
        for i in range(K):
            if topk_max[i] - topk_min[i] > 0:
                print(i, topk_max[i] - topk_min[i])
                print(idx1[0], index_nonzero[idx1[0]])
                print(idx2[0], index_zero[idx2[0]])
                mask_flatten[index_nonzero[idx1[0]]] = False
                mask_flatten[index_zero[idx2[0]]] = True
                print(mask_flatten)
                print(mask)

torch.manual_seed(1)
mask = nn.Parameter(torch.Tensor(2,3))
torch.nn.init.zeros_(mask)
mask.grad = torch.tensor(np.array([[1., 2., 3.], [4., 5., 6.]]), dtype=torch.float32)


with torch.no_grad():
    m = torch.Tensor(2,3).uniform_()
    m = m < 0.5
    print(m)
    mask.masked_fill_(m, 1)
print(mask)
updateScore(mask)
print(mask)
