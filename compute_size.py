#%%
import math
def compute_size(TT_dims, TT_ranks):
    N = len(TT_dims) // 2
    org_size = math.prod(TT_dims)
    size = 0
    for i, n in enumerate(TT_dims):
        print(f'core {i}: {TT_ranks[i]}, {n}, {TT_ranks[i+1]}')
        size += TT_ranks[i] * n * TT_ranks[i+1]
    return size, org_size

# 4 cores
TT_dims_att = [24,32,32,24]
TT_ranks_att = [1,24,30,24,1]
TT_dims_ffn = [32,24,48,64]
TT_ranks_ffn = [1,30,30,30,1]
size_att, org_size_att = compute_size(TT_dims_att, TT_ranks_att)
size_ffn, org_size_ffn = compute_size(TT_dims_ffn, TT_ranks_ffn)
size_enc = size_att * 4 + size_ffn * 2
org_size_enc = org_size_att * 4 + org_size_ffn * 2
print(org_size_enc, size_enc)
print(org_size_enc / size_enc)
### 6 cores base
# TT_dims_att = [12,8,8,8,8,12]
# TT_ranks_att = [1,12,64,64,64,12,1]
# TT_dims_ffn = [12,8,8,12,16,16]
# TT_ranks_ffn = [1,12,64,64,64,16,1]
### 6 cores large
# TT_dims_att = [16,8,8,8,8,16]
# TT_ranks_att = [1,16,96,96,96,16,1]
# TT_dims_ffn = [16,8,8,16,16,16]
# TT_ranks_ffn = [1,16,96,96,96,16,1]
### 8 cores large
TT_dims_att = [8,8,4,4,4,4,8,8]
TT_ranks_att = [1,8,64,96,96,96,64,8,1]
TT_dims_ffn = [8,8,4,4,8,8,8,8]
TT_ranks_ffn = [1,8,64,96,96,96,64,8,1]

size_att, org_size_att = compute_size(TT_dims_att, TT_ranks_att)
size_ffn, org_size_ffn = compute_size(TT_dims_ffn, TT_ranks_ffn)
size_enc = size_att * 4 + size_ffn * 2
org_size_enc = org_size_att * 4 + org_size_ffn * 2
print(org_size_enc, size_enc)
print(org_size_enc / size_enc)

#%%
from utils_tensor_layers import tensor_train
import torch

torch.manual_seed(42)

TT_dims = [12,8,8,8,8,12]
TT_ranks = [1,12,12*8,12*8*8,12*8,12,1]
W = torch.randn(*TT_dims)
factors = tensor_train(W, TT_ranks)

import tensorly as tl

# If your factors are a list of TT cores:
reconstructed = tl.tt_tensor.tt_to_tensor(factors)
print(torch.as_tensor(reconstructed)[0,0,0,0,0], '\n', W[0,0,0,0,0])
print(torch.allclose(W, torch.as_tensor(reconstructed), atol=1e-5))

#%%
from tensorly.decomposition import tensor_train

factors2 = tensor_train(W, TT_ranks)
# If your factors are a list of TT cores:
reconstructed2 = tl.tt_tensor.tt_to_tensor(factors2)
print(torch.as_tensor(reconstructed2)[0,0,0,0,0], '\n', W[0,0,0,0,0])
print(torch.allclose(W, torch.as_tensor(reconstructed2), atol=1e-5))

#%%