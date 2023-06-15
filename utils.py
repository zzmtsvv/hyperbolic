import torch
from torch.autograd import Function


class Atanh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=-1 + Atanh.eps, max=1 - Atanh.eps)
        ctx.save_for_backward(x)

        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 - (x * x)) if ctx.needs_input_grad[0] else None


class Asinh(Function):
    eps = 1e-5

    @staticmethod
    def forward(ctx, x):
        sqrt_x = torch.sqrt(x * x + 1)

        ctx.save_for_backward(sqrt_x)
        return torch.log(torch.clamp(sqrt_x + x, min=Asinh.eps))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / x if ctx.needs_input_grad[0] else None
