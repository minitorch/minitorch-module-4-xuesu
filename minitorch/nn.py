from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(
    input: Tensor, kernel: Tuple[int, int], pad_value: float
) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling
        pad_value: value to fillin padding section if height cannot be divided by kernal height or else wise

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert (
        height % kh == 0
    )  # don't know why minitorch simplify the pool and provide us the pad_value
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw
    _ans = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )
    return _ans


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape

    output, new_height, new_width = tile(input, kernel, 0.0)
    output = output.sum(4)
    output = output / (kernel[0] * kernel[1])
    output = output.view(batch, channel, new_height, new_width)
    return output


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        input, dim = ctx.saved_values
        return grad_output * argmax(input, int(dim.item())), 0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    exp_input = input.exp()
    mid = exp_input.sum(dim)
    output = exp_input / mid
    return output


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    max_input = max(input, dim)
    mid_input = input - max_input
    mid_input = mid_input.exp()
    mid_input = mid_input.sum(dim)
    mid_input = mid_input.log()
    lse_input = max_input + mid_input
    return input - lse_input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape

    output, new_height, new_width = tile(input, kernel, 0.0)
    output = max(output, 4)
    output = output.view(batch, channel, new_height, new_width)
    return output


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of keeping the value at each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    if ignore:
        return input
    rand_t = rand(input.shape, input.f)

    for i in range(len(rand_t._tensor._storage)):
        if rand_t._tensor._storage[i] >= rate:
            rand_t._tensor._storage[i] = 1.0
        else:
            rand_t._tensor._storage[i] = 0.0

    output = input * rand_t
    return output
