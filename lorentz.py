from math import sqrt
from typing import Optional, Tuple, Union
import torch
from manifold import Manifold
from utils import Acosh


acosh = Acosh.apply


class LorentzHyperboloid(Manifold):
    def __init__(self,
                 curvature: float = 1.0,
                 eps: float = 1e-5) -> None:
        super().__init__(curvature)

        self.curvature = curvature
        self.sqrt_c = sqrt(curvature)
        self.eps = eps
    
    def proj_(self,
              x: torch.Tensor,
              dim: Union[int, Tuple[int]] = -1) -> None:
        dn = x.size(dim) - 1
        left = torch.sqrt(
            self.curvature + torch.norm(x.narrow(dim, 1, dn), dim=dim) ** 2
        ).unsqueeze(dim)
        right = x.narrow(dim, 1, dn)
        proj = torch.cat([left, right], dim=dim)
        x.copy_(proj)
    
    def proj_vector(self,
                    x: torch.Tensor,
                    v: torch.Tensor,
                    dim: int = -1) -> torch.Tensor:
        inner = self.inner_product(x, v, keepdim=True, dim=dim)
        return v.addcmul(inner, x / self.curvature)

    @staticmethod
    def inner_product(u: torch.Tensor,
                      v: torch.Tensor,
                      keepdim: bool = False,
                      dim: int = -1) -> torch.Tensor:
        '''
            Minkowski inner product
        '''
        d = u.size(dim) - 1
        uv = u * v
        
        if keepdim:
            res = torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(dim=dim, keepdim=True)
            return res
        
        res = -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)
        return res
    
    def inner0(self,
               v: torch.Tensor,
               keepdim: bool = False,
               dim: int = -1) -> torch.Tensor:
        res = -v.narrow(dim, 0, 1) * self.sqrt_c
        if keepdim:
            return res
        return res.squeeze(dim)
    
    def dist0(self,
              x: torch.Tensor,
              keepdim: bool = False,
              dim: int = -1) -> torch.Tensor:
        d = self.inner0(x, keepdim=keepdim, dim=dim)
        return self.sqrt_c * acosh(d / self.curvature)
    
    def dist(self,
             u: torch.Tensor,
             v: torch.Tensor,
             keepdim: bool = False,
             dim: int = -1) -> torch.Tensor:
        d = self.inner_product(u, v, keepdim=keepdim, dim=dim)
        return self.sqrt_c * acosh(d / self.curvature)
    
    def zero_exp(self,
                 v: torch.Tensor,
                 dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        nomin = self.norm(v, keepdim=True, dim=dim)
        left = torch.cosh(nomin / self.sqrt_c) * self.sqrt_c
        right = self.sqrt_c * torch.sinh(nomin / self.sqrt_c) * v / nomin
        d = right.size(dim) - 1
        
        out = torch.cat((left + right.narrow(dim, 0, 1), right.narrow(dim, 1, d)), dim=dim)
        self.proj_(out, dim=dim)
        return out
    
    def exp(self,
            x: torch.Tensor,
            v: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        v = self.proj_vector(x, v, dim=dim)

        nomin = self.norm(v, keepdim=True, dim=dim)
        p = torch.cosh(nomin / self.sqrt_c) * x + self.sqrt_c * torch.sinh(nomin / self.sqrt_c) * v / nomin
        self.proj_(p, dim=dim)
        return p
    
    def zero_log(self,
                 x: torch.Tensor,
                 dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        dist = self.dist0(x, dim=dim, keepdim=True)
        nomin = 1.0 / self.curvature * self.inner0(x, keepdim=True) * self.sqrt_c
        d = x.size(dim) - 1
        nomin = torch.cat((nomin + x.narrow(dim, 0, 1), x.narrow(dim, 1, d)), dim)
        denominator = self.norm(nomin, keepdim=True)
        return dist * nomin / denominator
    
    def log(self,
            x: torch.Tensor,
            v: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        dist = self.dist(x, v, dim=dim, keepdim=True)
        nomin = v + 1.0 / self.curvature * self.inner_product(x, v, keepdim=True) * x
        denominator = self.norm(nomin, keepdim=True)
        return dist * nomin / denominator
    
    def parallel_transport(self,
                           x: torch.Tensor,
                           dim: Union[int, Tuple[int]] = -1,
                           from_: Optional[torch.Tensor] = None,
                           to_: Optional[torch.Tensor] = None) -> torch.Tensor:
        if from_ is None:
            return self.parallel_transport0(x, to_, dim)
        
        logmap = self.log(from_, to_, dim=dim)
        nomin = self.inner_product(logmap, x, keepdim=True)
        denominator = self.dist(from_, to_, dim=dim, keepdim=True) ** 2
        p = x - nomin / denominator * (logmap + self.log(to_, from_, dim=dim))
        return p

    def parallel_transport0(self,
                            x: torch.Tensor,
                            to: torch.Tensor,
                            dim: int = -1) -> torch.Tensor:
        logmap = self.zero_log(to, dim=dim)
        nomin = self.inner_product(logmap, x, keepdim=True)
        denominator = self.dist0(to, dim=dim, keepdim=True) ** 2
        
        p = x - nomin / denominator * (logmap + self.zero_log_back(to, dim=dim))
        return p
    
    def zero_log_back(self,
                      x: torch.Tensor,
                      dim: int = -1) -> torch.Tensor:
        dist = self.dist0(x, keepdim=True, dim=dim)
        nomin = 1.0 / self.curvature * self.inner0(x, keepdim=True) * x
        d = nomin.size(dim) - 1
        nomin = torch.cat(
            (nomin.narrow(dim, 0, 1) + self.sqrt_c, nomin.narrow(dim, 1, d)), dim=dim
        )
        denominator = self.norm(nomin, keepdim=True)
        return dist * nomin / denominator

    def norm(self,
             u: torch.Tensor,
             keepdim: bool = False,
             dim: int = -1) -> torch.Tensor:
        uu = self.inner_product(u, u, keepdim=keepdim, dim=dim)
        return torch.sqrt(torch.clamp(uu, min=self.eps))

    def __repr__(self) -> str:
        return super().__repr__()
    
    def __eq__(self, other) -> bool:
        return super().__eq__(other)


class GarinHyperboloid:
    # https://en.wikipedia.org/wiki/The_Garin_Death_Ray
    # kind of joke :)
    pass


if __name__ == "__main__":
    l = LorentzHyperboloid()
