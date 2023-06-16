from typing import Any
import torch
from torch.autograd import Function


class Atanh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx: Any, x: Any) -> Any:
        x = torch.clamp(x, min=-1 + Atanh.eps, max=1 - Atanh.eps)
        ctx.save_for_backward(x)

        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        x, = ctx.saved_tensors
        return grad_output / (1 - (x * x)) if ctx.needs_input_grad[0] else None


class Asinh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx: Any, x: Any) -> Any:
        sqrt_x = torch.sqrt(x * x + 1)

        ctx.save_for_backward(sqrt_x)
        return torch.log(torch.clamp(sqrt_x + x, min=Asinh.eps))

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        x, = ctx.saved_tensors
        return grad_output / x if ctx.needs_input_grad[0] else None


class Acosh(Function):
    eps = 1e-5
    
    @staticmethod
    def forward(ctx: Any, x: Any) -> Any:
        x = torch.clamp(x, min=1 + Acosh.eps)
        sqrt_x = torch.sqrt(x * x - 1)

        ctx.save_for_backward(sqrt_x)
        return torch.log(sqrt_x)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        x, = ctx.saved_tensors
        return grad_output / x if ctx.needs_input_grad[0] else None


def lorentz_2_poincare(x: torch.Tensor,
                       curvature: float,
                       dim: int = -1) -> torch.Tensor:
    '''
        Diffeomorphism maps Hyperboloid onto Poincare Ball

        x : Tensor - point on the Hyperboloid
        curvature: float - manifold curvature
        dim : int - reduction dimension for operations
    '''
    curvature = torch.as_tensor(curvature)
    d = x.size(dim) - 1
    return x.narrow(dim, 1, d) / (x.narrow(-dim, 0, 1) + torch.sqrt(curvature))


def poincare_2_lorentz(x: torch.Tensor,
                       curvature: float,
                       dim: int = -1,
                       eps: float = 1e-6) -> torch.Tensor:
    '''
        Diffeomorphism maps Poincare Ball onto Hyperboloid

        x : Tensor - point on the Poincare Ball
        curvature: float - manifold curvature
        dim : int - reduction dimension for operations
    '''
    curvature = torch.as_tensor(curvature)
    norm_squared = torch.sum(x * x, dim=dim, keepdim=True)
    res = torch.sqrt(curvature) * torch.cat((1 + norm_squared, 2 * x), dim=dim)
    return res / (1.0 - norm_squared + eps)
