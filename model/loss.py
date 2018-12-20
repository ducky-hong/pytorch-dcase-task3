import torch.nn.functional as F

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output.squeeze(), target.float())
