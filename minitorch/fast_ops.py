from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile functions with `njit`."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape) # type: ignore
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2])) # type: ignore
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
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
        # TODO: Implement for Task 3.1.
        raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
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
        # TODO: Implement for Task 3.1.
        out_size = 1
        n_dims = len(out_shape)
        for s in out_shape:
            out_size *= s

        strides_aligned = (
            out_strides == a_strides == b_strides and out_shape == a_shape == b_shape
        )

        if strides_aligned:
            for i in prange(out_size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            steps = np.empty(n_dims, dtype=np.int64)
            temp = 1
            for d in reversed(range(n_dims)):
                steps[d] = temp
                temp *= out_shape[d]

            for i in prange(out_size):
                a_index = 0
                b_index = 0
                for d in range(n_dims):
                    idx = (i // steps[d]) % out_shape[d]
                    idx_a = idx if a_shape[d] > 1 else 0
                    idx_b = idx if b_shape[d] > 1 else 0
                    a_index += idx_a * a_strides[d]
                    b_index += idx_b * b_strides[d]
                out[i] = fn(a_storage[a_index], b_storage[b_index])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

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
        # Number of dimensions
        n_dims = len(a_shape)

        # Compute the size of the output tensor
        out_size = 1
        for s in out_shape:
            out_size *= s

        # Precompute steps for index calculation
        out_strides_np = np.array(out_strides)
        a_strides_np = np.array(a_strides)
        a_shape_np = np.array(a_shape)
        out_shape_np = np.array(out_shape)

        # Prepare the shape for the reduced dimension
        reduce_dim_size = a_shape[reduce_dim]

        # Main loop in parallel over the output elements
        for out_index in prange(out_size):
            # Compute multi-dimensional index for output
            out_idx = np.empty(n_dims, dtype=np.int64)
            tmp = out_index
            for dim in range(n_dims - 1, -1, -1):
                out_idx[dim] = tmp % out_shape_np[dim]
                tmp = tmp // out_shape_np[dim]

            # Map output index to input index (excluding reduced dimension)
            a_idx = out_idx.copy()
            a_idx[reduce_dim] = 0  # Start at the first element in the reduced dimension

            # Compute the flat index for the first element
            a_flat_index = 0
            for dim in range(n_dims):
                a_flat_index += a_idx[dim] * a_strides_np[dim]

            # Initialize the accumulator with the first element
            acc = a_storage[a_flat_index]

            # Inner loop over the reduced dimension
            for rd in range(1, reduce_dim_size):
                # Update the index along the reduced dimension
                a_idx[reduce_dim] = rd
                # Compute the flat index
                a_flat_index = 0
                for dim in range(n_dims):
                    a_flat_index += a_idx[dim] * a_strides_np[dim]
                # Accumulate the result
                acc = fn(acc, a_storage[a_flat_index])

            # Compute the flat index for the output
            out_flat_index = 0
            for dim in range(n_dims):
                out_flat_index += out_idx[dim] * out_strides_np[dim]

            # Write the result to the output storage
            out[out_flat_index] = acc
    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
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
    """NUMBA tensor matrix multiply function."""
    # Precompute dimensions
    batch_size = max(a_shape[0], b_shape[0])
    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]

    # Precompute strides
    a_batch_stride = a_strides[0] if len(a_shape) == 3 else 0
    b_batch_stride = b_strides[0] if len(b_shape) == 3 else 0
    out_batch_stride = out_strides[0] if len(out_shape) == 3 else 0

    # Main loop in parallel over batches and output matrix elements
    for batch in prange(batch_size):
        for i in range(M):
            for j in range(N):
                # Compute initial indices
                out_index = batch * out_batch_stride + i * out_strides[-2] + j * out_strides[-1]
                a_index_base = (min(batch, a_shape[0]-1) * a_batch_stride +
                                i * a_strides[-2])
                b_index_base = (min(batch, b_shape[0]-1) * b_batch_stride +
                                j * b_strides[-1])
                # Initialize accumulator
                acc = 0.0
                # Inner loop over K dimension
                for k in range(K):
                    a_index = a_index_base + k * a_strides[-1]
                    b_index = b_index_base + k * b_strides[-2]
                    acc += a_storage[a_index] * b_storage[b_index]
                # Write the result to output storage
                out[out_index] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
