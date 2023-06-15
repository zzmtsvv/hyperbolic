from typing import Tuple, Union, Optional
from manifold import Manifold
import torch
from math import sqrt
from utils import Asinh, Atanh


atanh = Atanh.apply
asinh = Asinh.apply


class PoincareBall(Manifold):
    def __init__(self,
                 conformal_factor: float = 1.0,
                 eps: float = 1e-5) -> None:
        if conformal_factor == 0.0:
            raise TypeError("c=0 means we're using Euclidean Geometry. Try another value :)")
        
        self.conf_factor = conformal_factor
        self.sqrt_c = sqrt(conformal_factor)
        self.eps = eps
    
    def conformal_factor(self,
                         x: Optional[torch.Tensor] = None,
                         dim: Union[int, Tuple[int]] = -1,
                         keepdim: bool = False) -> Union[float, torch.Tensor]:
        if x is None:
            return 2.0
        
        return torch.clamp(
            2 / (1 - self.conf_factor * torch.sum(x * x, dim=dim, keepdim=keepdim)), min=self.eps
            )
    
    def clamp_inside_(self,
                      value: torch.Tensor,
                      min_: float,
                      max_: float):
        indexes = (value > min_) * (value < max_)

        if indexes.any():
            value[indexes] = self.eps * torch.sign(value[indexes])
    
    def hyperplane(self,
                   x: torch.Tensor,
                   p: torch.Tensor,
                   a: torch.Tensor) -> torch.Tensor:
        
        sum_ = self.add(self.mul(p, -1), x)
        sum_norm = torch.sum(sum_ * sum_, dim=-1)
        a_norm = torch.norm(a, dim=-1)

        denominator = (1 - self.conf_factor * sum_norm) * a_norm
        self.clamp_inside_(denominator, -self.eps, self.eps)

        tmp = self.conformal_factor(p) * a_norm / self.sqrt_c
        return tmp * asinh((2 * self.sqrt_c * torch.sum(sum_ * a, dim=-1)) / denominator)
    
    def linear(self,
               x: torch.Tensor,
               w: torch.Tensor) -> torch.Tensor:
        '''
            zero_log mapping + linear mapping + zero_exp mapping
        '''
        wx = x.matmul(w.t())
        wx_norm = torch.clamp(torch.norm(wx, dim=-1, keepdim=True), min=self.eps)
        x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=self.eps)

        tmp = (1 / self.sqrt_c) * torch.tanh(wx_norm * atanh(self.sqrt_c * x_norm) / x_norm)
        return tmp * wx / wx_norm
    
    def parallel_transport(self,
                           x: torch.Tensor,
                           dim: Union[int, Tuple[int]] = -1,
                           from_: Optional[torch.Tensor] = None,
                           to_: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x * self.conformal_factor(from_, dim=dim, keepdim=True) / self.conformal_factor(to_, dim=dim, keepdim=True)
    
    def zero_exp(self,
                 v: torch.Tensor,
                 dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point v from Tangent space at point 0 back to Manifold
        '''
        
        conf_vv = self.sqrt_c * torch.clamp(
            torch.norm(v, dim=dim, keepdim=True), min=self.eps
        )
        return torch.tanh(torch.clamp(conf_vv, min=-10, max=10)) * v / conf_vv
    
    def exp(self,
            x: torch.Tensor,
            v: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point v from Tangent space at point x back to Manifold
        '''
        c_vv = self.sqrt_c * torch.clamp(
            torch.norm(v, dim=dim, keepdim=True), min=self.eps
        )

        out = self.add(x,
                       torch.tanh(self.conformal_factor(x, dim=dim, keepdim=True) * c_vv / 2) * v / c_vv,
                       dim=dim)
        return out
    
    def zero_log(self,
                 x: torch.Tensor,
                 dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point y from Manifold to Tangent Space at point 0
        '''
        x_norm = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=self.eps)
        return (1 / self.sqrt_c) * atanh(self.sqrt_c * x_norm) * x / x_norm
    
    def log(self,
            x: torch.Tensor,
            v: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point v from Manifold to Tangent Space at point x
        '''
        sum_ = self.add(self.mul(x, -1, dim=dim), v, dim=dim)
        sum_norm = torch.clamp(torch.norm(sum_, dim=dim, keepdim=True), min=self.eps)

        c_factor = 2 / (self.sqrt_c * self.conformal_factor(x, dim=dim, keepdim=True))
        return c_factor * atanh(self.sqrt_c * sum_norm) * sum_ / sum_norm
    
    def mul(self,
            a: torch.Tensor,
            b: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        
        a_norm = torch.clamp(torch.norm(a, dim=dim, keepdim=True), min=self.eps)

        tmp = (1 / self.sqrt_c) * torch.tanh(torch.clamp(b * atanh(self.sqrt_c * a_norm), min=-10, max=10))
        return tmp * a / a_norm
    
    def add(self,
            x: torch.Tensor,
            y: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        
        c = self.conf_factor

        xy = torch.sum(x * y, dim=dim, keepdim=True)
        xx = torch.sum(x * x, dim=dim, keepdim=True)
        yy = torch.sum(y * y, dim=dim, keepdim=True)

        term1 = (1 + 2 * c * xy + c * yy) * x
        term2 = (1 - c * xx) * y
        denominator = 1 + 2 * c * xy + c * c * xx * yy

        self.clamp_inside_(denominator, -1e-12, 1e-12)

        return (term1 + term2) / denominator
    
    def proj_(self,
              x: torch.Tensor,
              dim: Union[int, Tuple[int]] = -1) -> None:
        with torch.no_grad():
            exp = self.zero_exp(x, dim=dim)
            x.copy_(exp)

    def __repr__(self) -> str:
        return "PoincareBallManifold, conformal_factor={}".format(self.conf_factor)
    
    def __eq__(self, other) -> bool:
        return self.conf_factor == other.conf_factor

