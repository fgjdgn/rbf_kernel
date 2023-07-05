import torch

def rbf_kernel(x, gamma):
    
    dist_sq = torch.sum(x ** 2)
    
    kernel_val = torch.exp(-gamma * dist_sq)
    return kernel_val

# usage
kernel_value = rbf_kernel(x, gamma)
