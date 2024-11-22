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
        # Check if strides and shapes are aligned                                              |
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):    |
            # If aligned, apply the function directly in parallel                              |
            for i in prange(len(out)):---------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                     |
        else:                                                                                  |
            # Handle misaligned strides/shapes with broadcasting and indexing                  |
            for i in prange(len(out)):---------------------------------------------------------| #1
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                                 |
                in_index = np.empty(MAX_DIMS, dtype=np.int32)                                  |
                                                                                               |
                to_index(i, out_shape, out_index)                                              |
                broadcast_index(out_index, out_shape, in_shape, in_index)                      |
                                                                                               |
                out_position = index_to_position(out_index, out_strides)                       |
                in_position = index_to_position(in_index, in_strides)                          |
                                                                                               |
                # Apply the function and store the result                                      |
                out[out_position] = fn(in_storage[in_position])                                |
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
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (180) is hoisted out
 of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (181) is hoisted out
 of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/dyl
anriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py
(218)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (218)
----------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                           |
        out: Storage,                                                                   |
        out_shape: Shape,                                                               |
        out_strides: Strides,                                                           |
        a_storage: Storage,                                                             |
        a_shape: Shape,                                                                 |
        a_strides: Strides,                                                             |
        b_storage: Storage,                                                             |
        b_shape: Shape,                                                                 |
        b_strides: Strides,                                                             |
    ) -> None:                                                                          |
        # Check if all strides and shapes are aligned                                   |
        strides_aligned = (                                                             |
            np.array_equal(out_strides, a_strides)                                      |
            and np.array_equal(out_strides, b_strides)                                  |
            and np.array_equal(out_shape, a_shape)                                      |
            and np.array_equal(out_shape, b_shape)                                      |
        )                                                                               |
                                                                                        |
        if strides_aligned:                                                             |
            # If aligned, apply the function directly in parallel                       |
            for i in prange(len(out)):--------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                 |
        else:                                                                           |
            # Handle misaligned strides/shapes with broadcasting and indexing           |
            for i in prange(len(out)):--------------------------------------------------| #3
                # Allocate index buffers                                                |
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                          |
                a_index = np.empty(MAX_DIMS, dtype=np.int32)                            |
                b_index = np.empty(MAX_DIMS, dtype=np.int32)                            |
                                                                                        |
                # Compute output, a, and b indices                                      |
                to_index(i, out_shape, out_index)                                       |
                broadcast_index(out_index, out_shape, a_shape, a_index)                 |
                broadcast_index(out_index, out_shape, b_shape, b_index)                 |
                                                                                        |
                # Compute positions based on strides                                    |
                out_position = index_to_position(out_index, out_strides)                |
                a_position = index_to_position(a_index, a_strides)                      |
                b_position = index_to_position(b_index, b_strides)                      |
                                                                                        |
                # Apply the function and store the result                               |
                out[out_position] = fn(a_storage[a_position], b_storage[b_position])    |
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
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (245) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (246) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (247) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/dylanriffle/Desktop/Cornell/ml_engineering/mod3-
djriffle/minitorch/fast_ops.py (286)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (286)
------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                    |
        out: Storage,                                                               |
        out_shape: Shape,                                                           |
        out_strides: Strides,                                                       |
        a_storage: Storage,                                                         |
        a_shape: Shape,                                                             |
        a_strides: Strides,                                                         |
        reduce_dim: int,                                                            |
    ) -> None:                                                                      |
        for i in prange(len(out)):--------------------------------------------------| #4
            # Allocate index buffers                                                |
            out_index = np.empty(MAX_DIMS, dtype=np.int32)                          |
            a_index = np.empty(MAX_DIMS, dtype=np.int32)                            |
                                                                                    |
            # Compute initial indices and positions                                 |
            to_index(i, out_shape, out_index)                                       |
            broadcast_index(out_index, out_shape, a_shape, a_index)                 |
            out_position = index_to_position(out_index, out_strides)                |
            a_position = index_to_position(a_index, a_strides)                      |
                                                                                    |
            # Initialize the output with the first element                          |
            out[out_position] = a_storage[a_position]                               |
                                                                                    |
            # Reduce along the specified dimension                                  |
            for j in range(1, a_shape[reduce_dim]):                                 |
                a_index[reduce_dim] = j                                             |
                a_position = index_to_position(a_index, a_strides)                  |
                out[out_position] = fn(out[out_position], a_storage[a_position])    |
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
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (297) is hoisted out
 of the parallel loop labelled #4 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/dylanriffle/Desktop
/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (298) is hoisted out
 of the parallel loop labelled #4 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```

# 3.2 - Matrix Multiply

```
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/dyla
nriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (318)

================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/dylanriffle/Desktop/Cornell/ml_engineering/mod3-djriffle/minitorch/fast_ops.py (318)
----------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                      |
    out: Storage,                                                                 |
    out_shape: Shape,                                                             |
    out_strides: Strides,                                                         |
    a_storage: Storage,                                                           |
    a_shape: Shape,                                                               |
    a_strides: Strides,                                                           |
    b_storage: Storage,                                                           |
    b_shape: Shape,                                                               |
    b_strides: Strides,                                                           |
) -> None:                                                                        |
    """NUMBA tensor matrix multiply function."""                                  |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                        |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                        |
                                                                                  |
    for batch in prange(out_shape[0]):--------------------------------------------| #5
        a_offset = batch * a_batch_stride                                         |
        b_offset = batch * b_batch_stride                                         |
                                                                                  |
        for row in range(a_shape[1]):                                             |
            for col in range(b_shape[2]):                                         |
                # Compute the dot product for the current row and column          |
                dot_product = 0.0                                                 |
                for k in range(a_shape[2]):                                       |
                    a_index = a_offset + row * a_strides[1] + k * a_strides[2]    |
                    b_index = b_offset + k * b_strides[1] + col * b_strides[2]    |
                    dot_product += a_storage[a_index] * b_storage[b_index]        |
                                                                                  |
                # Store the result in the output tensor                           |
                out_index = (                                                     |
                    batch * out_strides[0]                                        |
                    + row * out_strides[1]                                        |
                    + col * out_strides[2]                                        |
                )                                                                 |
                out[out_index] = dot_product                                      |
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

# 3.4

**Happy Thanksgiving**

![1732218119882](image/README/1732218119882.png)

## Results:

> Size: 64x64, CUDA Time: 0.040372s, Numba Time: 0.036536s
> Size: 128x128, CUDA Time: 0.149698s, Numba Time: 0.147405s
> Size: 256x256, CUDA Time: 0.580166s, Numba Time: 0.600350s
> Size: 512x512, CUDA Time: 2.331692s, Numba Time: 2.720708s
> Size: 1024x1024, CUDA Time: 9.240870s, Numba Time: 12.647923s
> Size: 2048x2048, CUDA Time: 37.108216s, Numba Time: 87.769194s

# 3.5

I updated the training script to display the average time per epoch as outlined in the assignment. Additionally, I modified the training process to implement early stopping if the model correctly labels all data for two consecutive epochs (to save time/google colab credits).

## GPU - Split Dataset - 100 Hidden Layer 0.05 Learning Rate

> Epochs 0-9: Average time per epoch: 2.204s
> Epoch  9  loss  4.346814027482488 correct 45
> Epochs 10-19: Average time per epoch: 1.971s
> Epoch  19  loss  2.157903632709191 correct 44
> Epochs 20-29: Average time per epoch: 1.964s
> Epoch  29  loss  2.8637940645805537 correct 44
> Epochs 30-39: Average time per epoch: 1.970s
> Epoch  39  loss  2.311212699289988 correct 45
> Epochs 40-49: Average time per epoch: 1.959s
> Epoch  49  loss  1.6140746723625266 correct 46
> Epochs 50-59: Average time per epoch: 1.960s
> Epoch  59  loss  1.4528505750483829 correct 48
> Epochs 60-69: Average time per epoch: 1.959s
> Epoch  69  loss  1.5174633545834295 correct 49
> Epochs 70-79: Average time per epoch: 1.958s
> Epoch  79  loss  1.4571803126219554 correct 49
> Epochs 80-89: Average time per epoch: 1.954s
> Epoch  89  loss  1.0148927688167007 correct 49
> Epochs 90-99: Average time per epoch: 1.971s
> Epoch  99  loss  0.8825420110021014 correct 49
> Epochs 100-109: Average time per epoch: 1.967s
> Epoch  109  loss  1.3937318068656754 correct 49
> Epoch  111  loss  3.1962381561136937 correct 50
> Epochs 110-119: Average time per epoch: 1.967s
> Epoch  119  loss  1.1106989833434944 correct 49
> Epochs 120-129: Average time per epoch: 1.968s
> Epoch  129  loss  0.44998530822370286 correct 49
> Epoch  132  loss  1.2194713084386628 correct 50
> Epoch  133  loss  3.1856292163517157 correct 50
> Early stopping triggered at epoch 133 (all predictions correct twice).
> Total average epoch time: 1.982s

## CPU - Split Dataset - 100 Hidden Layer 0.05 Learning Rate

> Epochs 0-9: Average time per epoch: 2.054s
> Epoch  9  loss  2.837253894409059 correct 40
> Epochs 10-19: Average time per epoch: 0.132s
> Epoch  19  loss  4.011155571481086 correct 46
> Epochs 20-29: Average time per epoch: 0.134s
> Epoch  29  loss  3.105287197866471 correct 47
> Epochs 30-39: Average time per epoch: 0.131s
> Epoch  39  loss  2.6195957371571423 correct 49
> Epochs 40-49: Average time per epoch: 0.131s
> Epoch  49  loss  3.238118578160422 correct 48
> Epoch  56  loss  2.924282458485897 correct 50
> Epochs 50-59: Average time per epoch: 0.131s
> Epoch  59  loss  2.093629508436419 correct 50
> Epoch  60  loss  2.2801129603813703 correct 50
> Early stopping triggered at epoch 60 (all predictions correct twice).
> Total average epoch time: 0.447s

## GPU - Simple Dataset - 100 Hidden Layer 0.05 Learning Rate

> Epochs 0-9: Average time per epoch: 2.319s
> Epoch  9  loss  1.6911248137726287 correct 48
> Epoch  17  loss  0.9900248050407188 correct 50
> Epoch  18  loss  2.2156966602335975 correct 50
> Early stopping triggered at epoch 18 (all predictions correct twice).
> Total average epoch time: 2.154s

## CPU - Simple Dataset - 100 Hidden Layer 0.05 Learning Rate

> Epochs 0-9: Average time per epoch: 2.062s
> Epoch  9  loss  2.609836980119164 correct 45
> Epochs 10-19: Average time per epoch: 0.134s
> Epoch  19  loss  1.656837894963247 correct 47
> Epoch  26  loss  1.7757434121994176 correct 50
> Epoch  27  loss  0.44285938044121503 correct 50
> Early stopping triggered at epoch 27 (all predictions correct twice).
> Total average epoch time: 0.822s

## GPU - Xor Dataset - 100 Hidden Layer 0.05 Learning Rate

> Epochs 0-9: Average time per epoch: 2.242s
> Epoch  9  loss  4.578101611377214 correct 37
> Epochs 10-19: Average time per epoch: 2.023s
> Epoch  19  loss  2.7805222924474364 correct 42
> Epochs 20-29: Average time per epoch: 2.004s
> Epoch  29  loss  3.7858912699253526 correct 46
> Epochs 30-39: Average time per epoch: 2.032s
> Epoch  39  loss  3.507771857784999 correct 47
> Epochs 40-49: Average time per epoch: 2.025s
> Epoch  49  loss  2.7764384046240203 correct 48
> Epochs 50-59: Average time per epoch: 2.019s
> Epoch  59  loss  2.8021203698550865 correct 48
> Epochs 60-69: Average time per epoch: 2.025s
> Epoch  69  loss  2.6926207218648077 correct 48
> Epochs 70-79: Average time per epoch: 2.021s
> Epoch  79  loss  2.4120885102857246 correct 48
> Epochs 80-89: Average time per epoch: 2.011s
> Epoch  89  loss  1.0299459950484677 correct 49
> Epochs 90-99: Average time per epoch: 2.021s
> Epoch  99  loss  0.9958882240532196 correct 48
> Epochs 100-109: Average time per epoch: 2.013s
> Epoch  109  loss  1.8871205102005018 correct 49
> Epochs 110-119: Average time per epoch: 2.009s
> Epoch  119  loss  0.5490348652496475 correct 49
> Epochs 120-129: Average time per epoch: 2.021s
> Epoch  129  loss  0.5610375179847171 correct 49
> Epochs 130-139: Average time per epoch: 2.013s
> Epoch  139  loss  2.018891541420463 correct 50
> Epoch  139  loss  2.018891541420463 correct 50
> Epoch  142  loss  1.5628729941633974 correct 50
> Epochs 140-149: Average time per epoch: 2.019s
> Epoch  149  loss  1.0384484637745564 correct 48
> Epoch  152  loss  1.4547089291418602 correct 50
> Epoch  153  loss  2.195140170574619 correct 50
> Epoch  153  loss  2.195140170574619 correct 50
> Early stopping triggered at epoch 153 (all predictions correct twice).
> Total average epoch time: 2.033s

## CPU - Xor Dataset - 100 Hidden Layer 0.05 Learning Rate

> Epochs 0-9: Average time per epoch: 2.080s
> Epoch  9  loss  5.517275021598561 correct 34
> Epochs 10-19: Average time per epoch: 0.133s
> Epoch  19  loss  5.379023762361035 correct 43
> Epochs 20-29: Average time per epoch: 0.133s
> Epoch  29  loss  6.442586800944841 correct 43
> Epochs 30-39: Average time per epoch: 0.131s
> Epoch  39  loss  3.7016348566336235 correct 43
> Epochs 40-49: Average time per epoch: 0.133s
> Epoch  49  loss  3.75750077413315 correct 43
> Epochs 50-59: Average time per epoch: 0.133s
> Epoch  59  loss  3.480886829373109 correct 44
> Epochs 60-69: Average time per epoch: 0.132s
> Epoch  69  loss  3.075690959319292 correct 45
> Epochs 70-79: Average time per epoch: 0.136s
> Epoch  79  loss  3.153016966152794 correct 44
> Epochs 80-89: Average time per epoch: 0.132s
> Epoch  89  loss  4.270194038316391 correct 44
> Epochs 90-99: Average time per epoch: 0.131s
> Epoch  99  loss  4.589162413558114 correct 46
> Epochs 100-109: Average time per epoch: 0.131s
> Epoch  109  loss  2.1739299956783253 correct 48
> Epochs 110-119: Average time per epoch: 0.131s
> Epoch  119  loss  2.1016534720040516 correct 45
> Epochs 120-129: Average time per epoch: 0.131s
> Epoch  129  loss  2.2175543268241467 correct 46
> Epochs 130-139: Average time per epoch: 0.132s
> Epoch  139  loss  1.6557675885537817 correct 47
> Epochs 140-149: Average time per epoch: 0.132s
> Epoch  149  loss  3.865399600543608 correct 48
> Epochs 150-159: Average time per epoch: 0.133s
> Epoch  159  loss  3.8784428295603384 correct 47
> Epochs 160-169: Average time per epoch: 0.133s
> Epoch  169  loss  2.685019863372572 correct 48
> Epochs 170-179: Average time per epoch: 0.133s
> Epoch  179  loss  1.4500403243594382 correct 48
> Epochs 180-189: Average time per epoch: 0.133s
> Epoch  189  loss  1.268838269687721 correct 49
> Epochs 190-199: Average time per epoch: 0.133s
> Epoch  199  loss  0.6633905676977141 correct 49
> Epochs 200-209: Average time per epoch: 0.131s
> Epoch  209  loss  1.6632230823342007 correct 49
> Epochs 210-219: Average time per epoch: 0.132s
> Epoch  219  loss  1.3102978209223803 correct 49
> Epochs 220-229: Average time per epoch: 0.132s
> Epoch  229  loss  0.8012066718599246 correct 49
> Epoch  236  loss  2.6446770830991766 correct 50
> Epochs 230-239: Average time per epoch: 0.131s
> Epoch  239  loss  2.7182279206932543 correct 49
> Epoch  244  loss  2.1350693279901534 correct 50
> Epochs 240-249: Average time per epoch: 0.132s
> Epoch  249  loss  2.009307011113557 correct 47
> Epoch  250  loss  0.8306906543945386 correct 50
> Epochs 250-259: Average time per epoch: 0.134s
> Epoch  259  loss  1.71187530074889 correct 49
> Epoch  263  loss  1.1499061343646668 correct 50
> Epoch  268  loss  1.442599895344542 correct 50
> Epochs 260-269: Average time per epoch: 0.134s
> Epoch  269  loss  0.9391960187764206 correct 49
> Epoch  273  loss  1.2575118448391702 correct 50
> Epoch  275  loss  1.799739377140986 correct 50
> Epoch  277  loss  0.4522503823418099 correct 50
> Epochs 270-279: Average time per epoch: 0.131s
> Epoch  279  loss  0.5531855287666504 correct 49
> Epoch  280  loss  0.6556600264806333 correct 50
> Epoch  281  loss  1.360674251524049 correct 50
> Epoch  281  loss  1.360674251524049 correct 50
> Early stopping triggered at epoch 281 (all predictions correct twice).
> Total average epoch time: 0.201s
