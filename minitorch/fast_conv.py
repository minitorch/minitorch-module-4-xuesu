from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Storage,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for i in prange(out_size):
        in_index = np.zeros(len(input_shape), dtype=np.int32)
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        to_index(i, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        in_index[:-2] = out_index[:-2]
        start_k = out_index[-1]
        tmp = 0.0
        for in_channel_now in range(in_channels):
            in_index[-2] = in_channel_now
            in_index[-1] = start_k
            in_pos_start = index_to_position(in_index, input_strides)
            for kwi in range(kw):
                kwi_used = kwi if not reverse else -kwi
                if 0 <= kwi_used + start_k < width:
                    weight_pos = (
                        out_index[1] * weight_strides[0]
                        + in_channel_now * weight_strides[1]
                        + kwi * weight_strides[2]
                    )

                    tmp += (
                        weight[weight_pos]
                        * input[in_pos_start + kwi_used * input_strides[2]]
                    )
        out[int(out_pos)] = tmp


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    kid = -1 if reverse else 1
    for i in prange(out_size):
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        to_index(i, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        batch_i, out_channel_i, h_start, w_start = out_index
        tmp = 0.0
        for in_channel_i in range(in_channels):
            for khi in range(kh):
                h_now = khi * kid + h_start
                if not (0 <= h_now < height):
                    break
                for kwi in range(kw):
                    w_now = kwi * kid + w_start
                    if not (0 <= w_now < width):
                        break
                    in_pos = (
                        batch_i * s10 + in_channel_i * s11 + h_now * s12 + w_now * s13
                    )
                    w_pos = (
                        out_channel_i * s20 + in_channel_i * s21 + khi * s22 + kwi * s23
                    )
                    # if debug:
                    #     print(reverse, i, out_pos,out_strides, "out[", batch_i, out_channel_i, h_start, w_start, "] += input[", batch_i, in_channel_i, h_now, w_now, "] * weight[", out_channel_i, in_channel_i, khi, kwi, "]", tmp, input[in_pos], weight[w_pos])
                    tmp += input[in_pos] * weight[w_pos]

        out[out_pos] = tmp


def _tensor_conv2d_matmul(
    input: Tensor,
    weight: Tensor,
    reverse: bool,
) -> Tensor:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        input (Tensor): input tensor, to be unroll
        weight (Tensor): weight tensor, to be permute
        reverse (bool): anchor weight at top-left or bottom-right

    Returns:
        output tensor
    """
    batch, in_channels, height, width = input.shape
    out_channels, in_channels_, kh, kw = weight.shape

    new_weight = (
        weight.contiguous().view(out_channels, in_channels * kh * kw).permute(1, 0)
    )
    new_input = input.zeros(shape=(batch, height, width, in_channels * kh * kw))
    olds0, olds1, olds2, olds3 = (
        input.strides[0],
        input.strides[1],
        input.strides[2],
        input.strides[3],
    )
    news0, news1, news2, news3 = (
        new_input.strides[0],
        new_input.strides[1],
        new_input.strides[2],
        new_input.strides[3],
    )
    khid = -1 if reverse else 1
    kwid = -1 if reverse else 1
    for batch_i in range(batch):
        for in_channel_i in range(in_channels):
            for hi in range(height):
                for wi in range(width):
                    for khi in range(kh):
                        h_now = khi * khid + hi
                        if not (0 <= h_now < height):
                            break
                        for kwi in range(kw):
                            w_now = kwi * kwid + wi
                            if not (0 <= w_now < width):
                                break
                            # print(batch_i, in_channel_i, hi, wi, khi, kwi, h_now, w_now)
                            in_pos_old = (
                                batch_i * olds0
                                + in_channel_i * olds1
                                + h_now * olds2
                                + w_now * olds3
                            )
                            in_pos_new = (
                                batch_i * news0
                                + hi * news1
                                + wi * news2
                                + (in_channel_i * kh * kw + khi * kw + kwi) * news3
                            )
                            new_input._tensor._storage[in_pos_new] = (
                                input._tensor._storage[in_pos_old]
                            )

    return (new_input @ new_weight).permute(0, 3, 1, 2)


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        # new_output = _tensor_conv2d_matmul(input, weight, reverse=False)
        # print("output", output, "new_output", new_output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
