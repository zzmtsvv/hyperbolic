from typing import Iterator
import torch
from torch import nn
from torch.optim.optimizer import required
from manifold import Manifold


class RSGD(torch.optim.SGD):
    '''
        Implements Riemannian Stochastic Gradient Descent.
    '''
    def __init__(self,
                 params: Iterator[nn.Parameter],
                 manifold: Manifold,
                 lr=required):
        super().__init__(params, lr)

        self.manifold = manifold
    
    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    lambda_square = self.manifold.conformal_factor(p, keepdim=True) ** 2
                    p.data.copy_(self.manifold.exp(p, -lr * p.grad.data / lambda_square))
