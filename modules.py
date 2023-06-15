from math import sqrt
from typing import Callable, Union, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from manifold import Manifold


class HyperbolicParameter(nn.Parameter):
    '''
        hyperbolic parameters and usual ones are supposed to be separated in order to optimize with different methods

        example:
            euclidean_optimizer = Adam(
                [p for p in model.parameters() if not isinstance(p, HyperbolicParameter)], ...
            )

            hyperbolic_optimizer = RSGD(
                [p for p in model.parameters() if isinstance(p, HyperbolicParameter)], ...
            )
    '''
    pass


class HyperbolicWrapper(nn.Module):
    def __init__(self,
                 f: Callable,
                 manifold: Manifold) -> None:
        super().__init__()

        self.func = f
        self.manifold = manifold
    
    def forward(self,
                x: torch.Tensor,
                dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        tangent = self.manifold.zero_log(x, dim)
        res = self.func(tangent)

        return self.manifold.zero_exp(res, dim)


class HyperbolicEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 manifold: Manifold,
                 padding_idx: Optional[int] = None,
                 scale_grad_by_freq: bool = False) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        
        self.padding_idx = padding_idx
        self.manifold = manifold
        self.scale_grad_by_freq = scale_grad_by_freq

        self.parameters()

        self.weight = HyperbolicParameter(torch.Tensor(num_embeddings, embedding_dim), requires_grad=True)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.uniform_(self.weight, -1e-3, 1e-3)
        self.manifold.proj_(self.weight)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.embedding(x, self.weight, self.padding_idx, scale_grad_by_freq=self.scale_grad_by_freq)
        return res


class Hyperplane(nn.Module):
    '''
        supposed to be the last layer in the model. (outputs logits in the appropriate hyperplane)
    '''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 manifold: Manifold) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self.p = HyperbolicParameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.uniform_(self.p, -1e-3, 1e-3)
        self.manifold.proj_(self.p)
        nn.init.uniform_(self.a, -1e-3, 1e-3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.size()

        res = x.unsqueeze(1).repeat(1, self.out_features, 1).view(-1, self.in_features)
        a = self.manifold.parallel_transport(self.a, to_=self.p)

        p = self.p.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, self.in_features)
        a = a.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, self.in_features)

        res = self.manifold.hyperplane(res, p, a)
        return res.view(batch_size, self.out_features)


class HyperbolicLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 manifold: Manifold,
                 bias: bool = True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

        if self.bias is not None:
            nn.init.uniform_(self.bias, -1e-3, 1e-3)
    
    def linear(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.linear(x, self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.linear(x)

        if self.bias is not None:
            return self.manifold.exp(res, self.manifold.parallel_transport(self.bias, to_=res))
        return res


