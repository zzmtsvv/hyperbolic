from typing import Tuple, Union, Optional
import torch
from abc import ABC, abstractmethod


class Manifold(ABC):
    @abstractmethod
    def proj_(self,
              x: torch.Tensor,
              dim: Union[int, Tuple[int]] = -1) -> None:
        raise NotImplementedError

    @abstractmethod
    def conformal_factor(self,
                         x: Optional[torch.Tensor] = None,
                         dim: Union[int, Tuple[int]] = -1,
                         keepdim: bool = False) -> Union[float, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def add(self,
            x: torch.Tensor,
            y: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def mul(self,
            a: torch.Tensor,
            b: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        raise NotImplementedError()

    def neg(self,
            x: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        return self.mul(x, -1, dim)

    @abstractmethod
    def log(self,
            x: torch.Tensor,
            v: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point v from Manifold to Tangent Space at point x
        '''
        raise NotImplementedError()

    @abstractmethod
    def exp(self,
            x: torch.Tensor,
            v: torch.Tensor,
            dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point v from Tangent space at point x back to Manifold
        '''
        raise NotImplementedError()

    @abstractmethod
    def zero_log(self,
                 x: torch.Tensor,
                 dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping of point x from Manifold to Tangent Space at point 0
        '''
        raise NotImplementedError()

    @abstractmethod
    def zero_exp(self,
                 v: torch.Tensor,
                 dim: Union[int, Tuple[int]] = -1) -> torch.Tensor:
        '''
            Mapping from Tangent space at point 0 of point v back to Manifold
        '''
        raise NotImplementedError()

    @abstractmethod
    def parallel_transport(self,
                           x: torch.Tensor,
                           dim: Union[int, Tuple[int]] = -1,
                           from_: Optional[torch.Tensor] = None,
                           to_: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def linear(self,
               x: torch.Tensor,
               w: torch.Tensor) -> torch.Tensor:
        '''
            zero_log mapping + linear mapping + zero_exp mapping
        '''
        raise NotImplementedError()

    @abstractmethod
    def hyperplane(self,
                   x: torch.Tensor,
                   p: torch.Tensor,
                   a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()
