# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/
* Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

    minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.1 - Basic Fast Ops

```
sum in tensor_functions.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/dyl
anriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py
(164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (164)
-----------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                  |
        out: Storage,                                                                          |
        out_shape: Shape,                                                                      |
        out_strides: Strides,                                                                  |
        in_storage: Storage,                                                                   |
        in_shape: Shape,                                                                       |
        in_strides: Strides,                                                                   |
    ) -> None:                                                                                 |
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):    |
            for i in prange(len(out)):---------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                     |
        else:                                                                                  |
            out_index = np.empty(MAX_DIMS, dtype=np.int32)                                     |
            in_index = np.empty(MAX_DIMS, dtype=np.int32)                                      |
            for i in prange(len(out)):---------------------------------------------------------| #1
                to_index(i, out_shape, out_index)                                              |
                broadcast_index(out_index, out_shape, in_shape, in_index)                      |
                                                                                               |
                o = index_to_position(out_index, out_strides)                                  |
                a = index_to_position(in_index, in_strides)                                    |
                                                                                               |
                out[o] = fn(in_storage[a])                                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/dyl
anriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py
(213)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (213)
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          |
        out: Storage,                                                  |
        out_shape: Shape,                                              |
        out_strides: Strides,                                          |
        a_storage: Storage,                                            |
        a_shape: Shape,                                                |
        a_strides: Strides,                                            |
        b_storage: Storage,                                            |
        b_shape: Shape,                                                |
        b_strides: Strides,                                            |
    ) -> None:                                                         |
        strides_aligned = (                                            |
            np.array_equal(out_strides, a_strides)                     |
            and np.array_equal(out_strides, b_strides)                 |
            and np.array_equal(out_shape, a_shape)                     |
            and np.array_equal(out_shape, b_shape)                     |
        )                                                              |
        if (strides_aligned):                                          |
            for i in prange(len(out)):---------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                |
        else:                                                          |
            out_idx = np.empty(MAX_DIMS, dtype=np.int32)               |
            a_idx = np.empty(MAX_DIMS, dtype=np.int32)                 |
            b_idx = np.empty(MAX_DIMS, dtype=np.int32)                 |
            for i in prange(len(out)):---------------------------------| #3
                                                                       |
                                                                       |
                to_index(i, out_shape, out_idx)                        |
                broadcast_index(out_idx, out_shape, a_shape, a_idx)    |
                broadcast_index(out_idx, out_shape, b_shape, b_idx)    |
                                                                       |
                o = index_to_position(out_idx, out_strides)            |
                a = index_to_position(a_idx, a_strides)                |
                b = index_to_position(b_idx, b_strides)                |
                out[o] = fn(a_storage[a], b_storage[b])                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/dylanriffle/Desktop/Cornell/ml_engineering/mod3-
djriffle/minitorch/fast_ops.py (273)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (273)
-----------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                   |
        out: Storage,                                                              |
        out_shape: Shape,                                                          |
        out_strides: Strides,                                                      |
        a_storage: Storage,                                                        |
        a_shape: Shape,                                                            |
        a_strides: Strides,                                                        |
        reduce_dim: int,                                                           |
    ) -> None:                                                                     |
        out_index = np.empty(MAX_DIMS, dtype=np.int32)                             |
        a_index = np.empty(MAX_DIMS, dtype=np.int32)                               |
        for i in prange(len(out)):-------------------------------------------------| #4
            to_index(i, out_shape, out_index)                                      |
            broadcast_index(out_index, out_shape, a_shape, a_index)                |
                                                                                   |
            o = index_to_position(out_index, out_strides)                          |
            a = index_to_position(a_index, a_strides)                              |
            out[o] = a_storage[a]                                                  |
                                                                                   |
            base_a_position = a - (a_index[reduce_dim] * a_strides[reduce_dim])    |
            for j in range(1, a_shape[reduce_dim]):                                |
                a_position = base_a_position + j * a_strides[reduce_dim]           |
                out[o] = fn(out[o], a_storage[a_position]) # type: ignore          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
# 3.2 - Matrix Multiply
```
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/dyla
nriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (300)

================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (300)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function."""                                          |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    for n in prange(out_shape[0]): -------------------------------------------------------| #5
        a_offset = n * a_batch_stride                                                     |
        b_offset = n * b_batch_stride                                                     |
                                                                                          |
        for i in range(a_shape[1]):                                                       |
            for j in range(b_shape[2]):                                                   |
                sum = 0.0                                                                 |
                for k in range(a_shape[2]):                                               |
                    a_idx = a_offset + i * a_strides[1] + k * a_strides[2]                |
                    b_idx = b_offset + k * b_strides[1] + j * b_strides[2]                |
                    sum += a_storage[a_idx] * b_storage[b_idx]                            |
                out_idx = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]    |
                out[out_idx] = sum                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
