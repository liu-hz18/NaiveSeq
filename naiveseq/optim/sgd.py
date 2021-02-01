import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.optimizers import FusedSGD as _FusedSGD
    has_fused_sgd = True

except ImportError:
    has_fused_sgd = False


def SGD(params, lr=1e-3, momentum=0., dampening=0., weight_decay=0, nesterov=False):
    """Implements stochastic gradient descent (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """
    if torch.cuda.is_available() and has_fused_sgd:
        return _FusedSGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    else:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
