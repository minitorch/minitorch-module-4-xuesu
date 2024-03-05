# from typing import Tuple

# import numpy as np
# import numba
# from numba import njit, prange, cuda

# from .autodiff import Context
# from .tensor import Tensor
# from .tensor_data import (
#     MAX_DIMS,
#     Index,
#     Shape,
#     Storage,
#     Strides,
#     to_index,
#     index_to_position,
#     broadcast_index
# )
# from .tensor_functions import Function

# to_index = cuda.jit(device=True)(to_index)
# index_to_position = cuda.jit(device=True)(index_to_position)
# broadcast_index = cuda.jit(device=True)(broadcast_index)


# def _tensor_conv1d(
#     out: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     input: Storage,
#     input_shape: Shape,
#     input_strides: Strides,
#     weight: Storage,
#     weight_shape: Shape,
#     weight_strides: Strides,
#     reverse: bool,
# ) -> None:
#     """
#     1D Convolution implementation.

#     Given input tensor of

#        `batch, in_channels, width`

#     and weight tensor

#        `out_channels, in_channels, k_width`

#     Computes padded output of

#        `batch, out_channels, width`

#     `reverse` decides if weight is anchored left (False) or right.
#     (See diagrams)

#     Args:
#         out (Storage): storage for `out` tensor.
#         out_shape (Shape): shape for `out` tensor.
#         out_strides (Strides): strides for `out` tensor.
#         out_size (int): size of the `out` tensor.
#         input (Storage): storage for `input` tensor.
#         input_shape (Shape): shape for `input` tensor.
#         input_strides (Strides): strides for `input` tensor.
#         weight (Storage): storage for `input` tensor.
#         weight_shape (Shape): shape for `input` tensor.
#         weight_strides (Strides): strides for `input` tensor.
#         reverse (bool): anchor weight at left or right
#     """
#     batch_, out_channels, out_width = out_shape
#     batch, in_channels, width = input_shape
#     out_channels_, in_channels_, kw = weight_shape

#     assert (
#         batch == batch_
#         and in_channels == in_channels_
#         and out_channels == out_channels_
#     )


#     out_size = 1
#     for sz in out_shape:
#         out_size *= sz
#     out_bbatch_stride = out_strides[-4] if len(out_strides) >= 4 else out_size
#     out_bbatch_size = out_size // out_bbatch_stride
#     # Batch dimension - fixed, c[batch, i, j]
#     batch = cuda.blockIdx.z
#     out_dim = len(out_shape)
#     a_dim = len(a_shape)
#     b_dim = len(b_shape)

#     BLOCK_DIM = 16
#     a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
#     b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
#     out_index = cuda.local.array(MAX_DIMS, numba.int32)
#     a_index = cuda.local.array(MAX_DIMS, numba.int32)
#     b_index = cuda.local.array(MAX_DIMS, numba.int32)

#     # The final position c[i, j]
#     i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

#     debug = False  # (cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z) == (2,2,0)
#     # The local position in the block.
#     pi = cuda.threadIdx.x
#     pj = cuda.threadIdx.y
#     # Code Plan:
#     # 1) Move across shared dimension by block dim.
#     #    a) Copy into shared memory for a matrix.
#     #    b) Copy into shared memory for b matrix
#     #    c) Compute the dot produce for position c[i, j]
#     dim_m = a_shape[-2]
#     dim_n = a_shape[-1]
#     dim_d = b_shape[-1]
#     if i >= dim_m and j >= dim_d:
#         return
#     out_pos_now = batch * out_strides[-3] + i * out_strides[-2] + j
#     for bbatch_i in range(out_bbatch_size):
#         to_index(
#             batch * out_strides[-3] + bbatch_i * out_bbatch_stride, out_shape, out_index
#         )
#         out_index[out_dim - 2] = i
#         out_index[out_dim - 1] = j
#         broadcast_index(out_index, out_shape, a_shape, a_index)
#         broadcast_index(out_index, out_shape, b_shape, b_index)
#         tmp = 0.0
#         for base_n in range(0, dim_n, BLOCK_DIM):
#             a_index[a_dim - 1] = base_n + pj
#             a_c = b_c = 0

#             if a_index[a_dim - 1] < dim_n and i < dim_m:
#                 a_c = 344
#                 a_pos = index_to_position(a_index, a_strides)
#                 a_shared[pi, pj] = a_storage[a_pos]
#                 if debug:
#                     print(
#                         cuda.blockIdx.x,
#                         cuda.blockIdx.y,
#                         cuda.blockIdx.z,
#                         out_pos_now,
#                         base_n,
#                         i,
#                         j,
#                         "a_shared",
#                         pi,
#                         pj,
#                         a_index[a_dim - 2],
#                         a_index[a_dim - 1],
#                         a_pos,
#                         a_storage[a_pos],
#                     )

#             b_index[b_dim - 2] = base_n + pi
#             if b_index[b_dim - 2] < dim_n and j < dim_d:
#                 b_c = 234
#                 b_pos = index_to_position(b_index, b_strides)
#                 b_shared[pi, pj] = b_storage[b_pos]
#                 if debug:
#                     print(
#                         cuda.blockIdx.x,
#                         cuda.blockIdx.y,
#                         cuda.blockIdx.z,
#                         out_pos_now,
#                         base_n,
#                         i,
#                         j,
#                         "b_shared",
#                         pi,
#                         pj,
#                         b_index[b_dim - 2],
#                         b_index[b_dim - 1],
#                         b_pos,
#                     )
#             if debug:
#                 print(
#                     "numba.cuda.syncthreads, ",
#                     bbatch_i,
#                     cuda.blockIdx.x,
#                     cuda.blockIdx.y,
#                     cuda.blockIdx.z,
#                     cuda.blockDim.x,
#                     cuda.blockDim.y,
#                     cuda.blockDim.z,
#                     out_pos_now,
#                     base_n,
#                     i,
#                     j,
#                 )
#             numba.cuda.syncthreads()
#             if i < dim_m and j < dim_d:
#                 k_lim = min(BLOCK_DIM, dim_n - base_n)
#                 for k in range(k_lim):
#                     if debug:
#                         print(
#                             cuda.blockIdx.x,
#                             cuda.blockIdx.y,
#                             cuda.blockIdx.z,
#                             out_pos_now,
#                             "out",
#                             i,
#                             j,
#                             "add",
#                             "a",
#                             pi,
#                             k,
#                             "b",
#                             k,
#                             pj,
#                             a_shared[pi, k],
#                             b_shared[k, pj],
#                             a_c,
#                             b_c,
#                         )
#                     tmp += a_shared[pi, k] * b_shared[k, pj]
#             numba.cuda.syncthreads()  # Note:!!!!!!!!!!!!!

#         if i < dim_m and j < dim_d and out_pos_now < out_size:
#             out[out_pos_now] = tmp
#         out_pos_now += out_bbatch_stride


# tensor_conv1d = cuda.jit()(_tensor_conv1d)


# class Conv1dFun(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         """
#         Compute a 1D Convolution

#         Args:
#             ctx : Context
#             input : batch x in_channel x h x w
#             weight : out_channel x in_channel x kh x kw

#         Returns:
#             batch x out_channel x h x w
#         """
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, w = input.shape
#         out_channels, in_channels2, kw = weight.shape
#         assert in_channels == in_channels2

#         # Run convolution
#         output = input.zeros((batch, out_channels, w))
#         tensor_conv1d(
#             *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
#         )
#         return output

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         input, weight = ctx.saved_values
#         batch, in_channels, w = input.shape
#         out_channels, in_channels, kw = weight.shape
#         grad_weight = grad_output.zeros((in_channels, out_channels, kw))
#         new_input = input.permute(1, 0, 2)
#         new_grad_output = grad_output.permute(1, 0, 2)
#         tensor_conv1d(
#             *grad_weight.tuple(),
#             grad_weight.size,
#             *new_input.tuple(),
#             *new_grad_output.tuple(),
#             False,
#         )
#         grad_weight = grad_weight.permute(1, 0, 2)

#         grad_input = input.zeros((batch, in_channels, w))
#         new_weight = weight.permute(1, 0, 2)
#         tensor_conv1d(
#             *grad_input.tuple(),
#             grad_input.size,
#             *grad_output.tuple(),
#             *new_weight.tuple(),
#             True,
#         )
#         return grad_input, grad_weight


# conv1d = Conv1dFun.apply


# def _tensor_conv2d(
#     out: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     input: Storage,
#     input_shape: Shape,
#     input_strides: Strides,
#     weight: Storage,
#     weight_shape: Shape,
#     weight_strides: Strides,
#     reverse: bool,
# ) -> None:
#     """
#     2D Convolution implementation.

#     Given input tensor of

#        `batch, in_channels, height, width`

#     and weight tensor

#        `out_channels, in_channels, k_height, k_width`

#     Computes padded output of

#        `batch, out_channels, height, width`

#     `Reverse` decides if weight is anchored top-left (False) or bottom-right.
#     (See diagrams)


#     Args:
#         out (Storage): storage for `out` tensor.
#         out_shape (Shape): shape for `out` tensor.
#         out_strides (Strides): strides for `out` tensor.
#         out_size (int): size of the `out` tensor.
#         input (Storage): storage for `input` tensor.
#         input_shape (Shape): shape for `input` tensor.
#         input_strides (Strides): strides for `input` tensor.
#         weight (Storage): storage for `input` tensor.
#         weight_shape (Shape): shape for `input` tensor.
#         weight_strides (Strides): strides for `input` tensor.
#         reverse (bool): anchor weight at top-left or bottom-right
#     """
#     batch_, out_channels, _, _ = out_shape
#     batch, in_channels, height, width = input_shape
#     out_channels_, in_channels_, kh, kw = weight_shape

#     assert (
#         batch == batch_
#         and in_channels == in_channels_
#         and out_channels == out_channels_
#     )

#     s1 = input_strides
#     s2 = weight_strides
#     # inners
#     s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
#     s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

#     khid = -1 if reverse else 1
#     kwid = -1 if reverse else 1
#     for i in prange(out_size):
#         out_index = np.zeros(len(out_shape), dtype=np.int32)
#         to_index(i, out_shape, out_index)
#         out_pos = index_to_position(out_index, out_strides)
#         batch_i, out_channel_i, h_start, w_start = out_index
#         tmp = 0.0
#         for in_channel_i in range(in_channels):
#             for khi in range(kh):
#                 h_now = khi * khid + h_start
#                 if not (0 <= h_now < height):
#                     break
#                 for kwi in range(kw):
#                     w_now = kwi * kwid + w_start
#                     if not (0 <= w_now < width):
#                         break
#                     in_pos = (
#                         batch_i * s10 + in_channel_i * s11 + h_now * s12 + w_now * s13
#                     )
#                     w_pos = (
#                         out_channel_i * s20 + in_channel_i * s21 + khi * s22 + kwi * s23
#                     )
#                     # if debug:
#                     #     print(reverse, i, out_pos,out_strides, "out[", batch_i, out_channel_i, h_start, w_start, "] += input[", batch_i, in_channel_i, h_now, w_now, "] * weight[", out_channel_i, in_channel_i, khi, kwi, "]", tmp, input[in_pos], weight[w_pos])
#                     tmp += input[in_pos] * weight[w_pos]
#         out[out_pos] = tmp


# tensor_conv2d = cuda.jit()(_tensor_conv2d)


# class Conv2dFun(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         """
#         Compute a 2D Convolution

#         Args:
#             ctx : Context
#             input : batch x in_channel x h x w
#             weight  : out_channel x in_channel x kh x kw

#         Returns:
#             (:class:`Tensor`) : batch x out_channel x h x w
#         """
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, h, w = input.shape
#         out_channels, in_channels2, kh, kw = weight.shape
#         assert in_channels == in_channels2
#         output = input.zeros((batch, out_channels, h, w))
#         tensor_conv2d(
#             *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
#         )
#         # new_output = _tensor_conv2d_matmul(input, weight, reverse=False)
#         # print("output", output, "new_output", new_output)
#         return output

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         input, weight = ctx.saved_values
#         batch, in_channels, h, w = input.shape
#         out_channels, in_channels, kh, kw = weight.shape

#         grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
#         new_input = input.permute(1, 0, 2, 3)
#         new_grad_output = grad_output.permute(1, 0, 2, 3)
#         tensor_conv2d(
#             *grad_weight.tuple(),
#             grad_weight.size,
#             *new_input.tuple(),
#             *new_grad_output.tuple(),
#             False,
#         )
#         grad_weight = grad_weight.permute(1, 0, 2, 3)

#         grad_input = input.zeros((batch, in_channels, h, w))
#         new_weight = weight.permute(1, 0, 2, 3)
#         tensor_conv2d(
#             *grad_input.tuple(),
#             grad_input.size,
#             *grad_output.tuple(),
#             *new_weight.tuple(),
#             True,
#         )
#         return grad_input, grad_weight


# conv2d = Conv2dFun.apply
