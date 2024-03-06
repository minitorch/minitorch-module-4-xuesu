from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    IndexingError,
    UserShape,
    UserStrides,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


def _check_shape_able_to_used_in_big_pos2small_pos(
    big_shape: Shape, small_shape: Shape
) -> None:
    assert np.prod(big_shape) >= np.prod(small_shape)
    assert (
        list(shape_broadcast(big_shape.tolist(), small_shape.tolist()))
        == big_shape.tolist()
    )


def _big_pos2small_pos(
    big_pos: int,
    big_shape: Shape,
    big_strides: Strides,
    small_shape: Shape,
    small_strides: Strides,
) -> int:
    need_broadcast = False
    if not np.array_equal(small_shape, big_shape) or not np.array_equal(
        big_strides, small_strides
    ):
        need_broadcast = True

    big_index = np.zeros(len(big_shape), dtype=np.int32)
    to_index(big_pos, big_shape, big_index)
    if need_broadcast:
        small_index = np.zeros(len(small_shape), dtype=np.int32)
        broadcast_index(big_index, big_shape, small_shape, small_index)
    else:
        small_index = big_index
    small_pos = index_to_position(small_index, small_strides)
    assert big_pos == index_to_position(big_index, big_strides)
    assert small_pos == index_to_position(small_index, small_strides)
    return small_pos


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor: ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:  # type: ignore[empty-body]
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # type: ignore[empty-body]
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # type: ignore[empty-body]
        pass

    @staticmethod
    def reduce(  # type: ignore[empty-body]
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:  # type: ignore[empty-body]
        pass

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        A collection of tensor functions
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.

        Returns:
            The wrapped function

        Fn:
            Args:
                a (:class:`TensorData`): tensor to map over
                out (:class:`TensorData`): optional, tensor data to fill in,
                    should broadcast with `a`

            Returns:
                new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply

        Returns:
            The wrapped function

        Fn:
            Args:
                a (:class:`TensorData`): tensor to zip over
                b (:class:`TensorData`): tensor to zip over

            Returns:
                :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])

        Args:
            fn: function from two floats-to-float to apply
            start: start value t[0] = fn(t[n], fn(t[n-1], ...fn(t[0], start)))

        Returns:
            The wrapped function

        Fn:
            Args:
                a (:class:`TensorData`): tensor to reduce over
                dim (int): int of dim to reduce

            Returns:
                :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        from minitorch.fast_ops import FastOps

        return FastOps.matrix_multiply(a, b)

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        _check_shape_able_to_used_in_big_pos2small_pos(out_shape, in_shape)
        mid_storage = [fn(v) for v in in_storage]
        for out_pos in range(len(out)):
            in_pos = _big_pos2small_pos(
                out_pos, out_shape, out_strides, in_shape, in_strides
            )
            out[out_pos] = mid_storage[in_pos]

    return _map


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        _check_shape_able_to_used_in_big_pos2small_pos(out_shape, a_shape)
        _check_shape_able_to_used_in_big_pos2small_pos(out_shape, b_shape)
        for out_pos in range(len(out)):
            a_pos = _big_pos2small_pos(
                out_pos, out_shape, out_strides, a_shape, a_strides
            )
            b_pos = _big_pos2small_pos(
                out_pos, out_shape, out_strides, b_shape, b_strides
            )
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float

    Returns:
        Tensor reduce function.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for out_pos in range(len(out)):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(out_pos, out_shape, out_index)
            assert out_index[reduce_dim] == 0
            a_index = np.copy(out_index)
            ans = out[out_pos]
            for i in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = i
                a_pos = index_to_position(a_index, a_strides)
                ans = fn(a_storage[a_pos], ans)
            out[out_pos] = ans

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
