# WowML — 14-Day Full-Scope Engineering Masterplan

> **Status**: Core C11 + OpenMP kernel, ctypes zero-copy bridge, Autograd engine, Linalg, NN, Optim, Data, Binary Serialization, and C/Python test suites implemented and passing (`4/4` CTest, `25/25` Pytest). Remaining items: `SVD` in `linalg_ops.py`, linking OpenBLAS/LAPACKE for CPU parity in benchmarks, and wiring `examples/*.py`.
> **Constraint**: Zero scope cuts. All layers (C11 + BLAS kernel, autograd, linalg, nn, optim, data, utils, benchmarks) are included.

---

## Day 1: Memory Subsystem, SIMD Alignment, and Error Handling

- [x] **1.1 Status Codes & Error Assertion Macros**
  - Path: [`csrc/include/wowml/error.h`](file:///c:/vs_code/wowml/csrc/include/wowml/error.h), [`csrc/src/error.c`](file:///c:/vs_code/wowml/csrc/src/error.c)
  - Tasks: Implement `wowml_status_string()` for all codes including `WOWML_ERR_NOT_IMPLEMENTED`. Verify `WOWML_CHECK(expr, status)` macro.
  - Acceptance Criteria: Invalid argument checks correctly return appropriate error codes without aborting.

- [x] **1.2 SIMD 64-Byte Aligned Allocation**
  - Path: [`csrc/include/wowml/memory.h`](file:///c:/vs_code/wowml/csrc/include/wowml/memory.h), [`csrc/src/memory.c`](file:///c:/vs_code/wowml/csrc/src/memory.c)
  - Tasks: Implement `wowml_aligned_alloc` using `_aligned_malloc` on Windows and `posix_memalign` on POSIX. Implement matching `wowml_aligned_free`.
  - Acceptance Criteria: Returned pointers are 64-byte aligned (`(uintptr_t)ptr % 64 == 0`).

- [x] **1.3 Memory Arena / Scratch Allocator**
  - Path: [`csrc/include/wowml/memory.h`](file:///c:/vs_code/wowml/csrc/include/wowml/memory.h), [`csrc/src/memory.c`](file:///c:/vs_code/wowml/csrc/src/memory.c)
  - Tasks: Implement `wowml_arena_init`, `wowml_arena_alloc` (aligned bump pointer), `wowml_arena_reset`, and `wowml_arena_destroy`.
  - Acceptance Criteria: Temporary allocations run without calling global heap malloc in inner loops.

---

## Day 2: Tensor Lifecycle, N-D Strides, and Zero-Copy Views

- [x] **2.1 Contiguous Tensor Allocation**
  - Path: [`csrc/include/wowml/tensor.h`](file:///c:/vs_code/wowml/csrc/include/wowml/tensor.h), [`csrc/src/tensor.c`](file:///c:/vs_code/wowml/csrc/src/tensor.c)
  - Tasks: Implement `wowml_tensor_alloc`, `wowml_tensor_free`, `wowml_tensor_from_data`, `wowml_tensor_zeros`, `wowml_tensor_ones`, and `wowml_tensor_fill`.
  - Acceptance Criteria: Correct row-major stride calculations for 1D through 8D tensors; `owns_data` respected on free.

- [x] **2.2 Tensor Cloning & Flat Accessors**
  - Path: [`csrc/src/tensor.c`](file:///c:/vs_code/wowml/csrc/src/tensor.c)
  - Tasks: Implement `wowml_tensor_clone`, `wowml_tensor_set_flat`, and `wowml_tensor_get_flat`.
  - Acceptance Criteria: Deep copy creates an independent buffer; flat get/set operates across all supported dtypes.

- [x] **2.3 Zero-Copy Reshape and Transposition**
  - Path: [`csrc/src/tensor.c`](file:///c:/vs_code/wowml/csrc/src/tensor.c)
  - Tasks: Implement `wowml_tensor_is_contiguous`, `wowml_tensor_contiguous` (buffer compaction), `wowml_tensor_reshape`, and `wowml_tensor_transpose`.
  - Acceptance Criteria: `transpose` swaps strides without copying data; `contiguous` creates a packed copy only when needed.

---

## Day 3: N-D Slicing and Generalized Broadcasting Engine

- [x] **3.1 Zero-Copy Strided Slicing**
  - Path: [`csrc/src/tensor.c`](file:///c:/vs_code/wowml/csrc/src/tensor.c)
  - Tasks: Implement `wowml_tensor_slice(view, src, dim, start, end, step)`.
  - Acceptance Criteria: Modifies `view->data` pointer offset and `view->strides[dim]` without allocating a new buffer; `owns_data` is 0.

- [x] **3.2 NumPy-Style Shape Broadcasting**
  - Path: [`csrc/src/tensor.c`](file:///c:/vs_code/wowml/csrc/src/tensor.c)
  - Tasks: Implement `wowml_broadcast_shapes` and `wowml_tensor_broadcast_to`.
  - Acceptance Criteria: Supports prepending 1-dimensions and expanding dimensions with stride 0.

- [x] **3.3 Tensor Test Verification**
  - Path: [`csrc/tests/test_tensor.c`](file:///c:/vs_code/wowml/csrc/tests/test_tensor.c)
  - Tasks: Build and run `ctest --output-on-failure -R test_tensor`.
  - Acceptance Criteria: All tensor allocation, slicing, view, and broadcast checks pass.

---

## Day 4: Element-Wise Math, Reductions, and Unary Operators

- [x] **4.1 Broadcasted Element-Wise Arithmetic**
  - Path: [`csrc/include/wowml/ops.h`](file:///c:/vs_code/wowml/csrc/include/wowml/ops.h), [`csrc/src/ops.c`](file:///c:/vs_code/wowml/csrc/src/ops.c)
  - Tasks: Implement `wowml_add`, `wowml_sub`, `wowml_mul`, and `wowml_div` with broadcasting and dtype dispatch.
  - Acceptance Criteria: Operations handle scalar, same-shape, and broadcasted inputs with zero buffer leaks.

- [x] **4.2 Unary Mathematical Operations**
  - Path: [`csrc/src/ops.c`](file:///c:/vs_code/wowml/csrc/src/ops.c)
  - Tasks: Implement `wowml_neg`, `wowml_exp`, `wowml_log`, `wowml_relu`, `wowml_sigmoid`, `wowml_tanh_op`, `wowml_sqrt`, and `wowml_pow_op`.
  - Acceptance Criteria: Correct element-wise numerical outputs matching standard IEEE math.

- [x] **4.3 Multi-Dimensional Reductions**
  - Path: [`csrc/src/ops.c`](file:///c:/vs_code/wowml/csrc/src/ops.c)
  - Tasks: Implement `wowml_sum`, `wowml_mean`, `wowml_max`, and `wowml_min` with `axis` and `keepdims` support.
  - Acceptance Criteria: Axis-specific and global reductions match NumPy reference outputs.

- [x] **4.4 Ops Test Harness**
  - Path: [`csrc/tests/test_ops.c`](file:///c:/vs_code/wowml/csrc/tests/test_ops.c)
  - Tasks: Run `ctest --output-on-failure -R test_ops`.
  - Acceptance Criteria: Passes arithmetic, unary math, and reduction test cases.

---

## Day 5: Matrix Multiplication, OpenMP Tiling, and BLAS Dispatch

- [x] **5.1 Pure C11 Blocked MatMul**
  - Path: [`csrc/include/wowml/matmul.h`](file:///c:/vs_code/wowml/csrc/include/wowml/matmul.h), [`csrc/src/matmul.c`](file:///c:/vs_code/wowml/csrc/src/matmul.c)
  - Tasks: Implement `wowml_matmul_c` using cache-friendly loop blocking (e.g., $64 \times 64$ tiles) and OpenMP `#pragma omp parallel for`.
  - Acceptance Criteria: Correct results for arbitrary matrix sizes $(M, K) \times (K, N)$.

- [x] **5.2 OpenBLAS Fast Path Integration**
  - Path: [`csrc/src/matmul.c`](file:///c:/vs_code/wowml/csrc/src/matmul.c)
  - Tasks: Implement `wowml_matmul_blas` calling `cblas_dgemm` / `cblas_sgemm` when compiled with `-DWOWML_USE_BLAS=1`.
  - Acceptance Criteria: Automatic fallback to pure C11 kernel when BLAS is disabled or not installed.

- [x] **5.3 Batched Matmul (`bmm`), Matrix-Vector, and Dot Product**
  - Path: [`csrc/src/matmul.c`](file:///c:/vs_code/wowml/csrc/src/matmul.c)
  - Tasks: Implement `wowml_bmm` for 3D tensors $(B, M, K) \times (B, K, N)$, `wowml_matvec`, and `wowml_dot`.
  - Acceptance Criteria: High-throughput batch matrix multiplication verified via [`csrc/tests/test_matmul.c`](file:///c:/vs_code/wowml/csrc/tests/test_matmul.c).

---

## Day 6: Linear Algebra Decompositions and System Solvers

- [x] **6.1 LU Decomposition, Determinant, and Matrix Inversion**
  - Path: [`csrc/include/wowml/linalg.h`](file:///c:/vs_code/wowml/csrc/include/wowml/linalg.h), [`csrc/src/linalg.c`](file:///c:/vs_code/wowml/csrc/src/linalg.c)
  - Tasks: Implement `wowml_linalg_lu` (Doolittle with partial pivoting), `wowml_linalg_det`, and `wowml_linalg_inv`.
  - Acceptance Criteria: Reconstructs $PA = LU$; accurately inverts non-singular square matrices.

- [x] **6.2 Linear System Solver (`solve`)**
  - Path: [`csrc/src/linalg.c`](file:///c:/vs_code/wowml/csrc/src/linalg.c)
  - Tasks: Implement `wowml_linalg_solve` for $Ax = b$ via forward and backward substitution.
  - Acceptance Criteria: $\|Ax - b\| < 10^{-6}$.

- [x] **6.3 QR, Cholesky, Eigenvalues (`eigh`), and SVD**
  - Path: [`csrc/src/linalg.c`](file:///c:/vs_code/wowml/csrc/src/linalg.c)
  - Tasks: Implement Modified Gram-Schmidt `qr`, Cholesky-Banachiewicz `cholesky`, Jacobi `eigh`, and `svd`. (LAPACKE fast paths optional when `WOWML_USE_BLAS` is enabled.)
  - Acceptance Criteria: Verified via [`csrc/tests/test_linalg.c`](file:///c:/vs_code/wowml/csrc/tests/test_linalg.c) and `tests/test_linalg.py` against NumPy.

---

## Day 7: Fused Neural Network Kernels & im2col / col2im

- [x] **7.1 Fused Linear + Bias + ReLU Kernel**
  - Path: [`csrc/include/wowml/ops.h`](file:///c:/vs_code/wowml/csrc/include/wowml/ops.h), [`csrc/src/ops.c`](file:///c:/vs_code/wowml/csrc/src/ops.c)
  - Tasks: Implement `wowml_fused_linear_bias_relu(out, x, weight, bias)`.
  - Acceptance Criteria: Executes $x W^T + b$ and $\max(0, \cdot)$ in a single memory pass without intermediate allocation.

- [x] **7.2 Numerically Stable Fused Cross Entropy**
  - Path: [`csrc/src/ops.c`](file:///c:/vs_code/wowml/csrc/src/ops.c)
  - Tasks: Implement `wowml_fused_cross_entropy` computing loss and softmax gradients simultaneously.
  - Acceptance Criteria: Uses max-subtraction trick to avoid floating-point overflow.

- [x] **7.3 2D Convolution Helpers: `im2col` & `col2im`**
  - Path: [`csrc/src/ops.c`](file:///c:/vs_code/wowml/csrc/src/ops.c)
  - Tasks: Implement `wowml_im2col` and backward `wowml_col2im` with support for padding, strides, and dilations.
  - Acceptance Criteria: Correct column matrix representation of image patches.

---

## Day 8: ctypes Python Bridge & Zero-Copy Memory Interoperability

- [x] **8.1 ctypes Function & Struct Signatures**
  - Path: [`wowml/_C/_bindings.py`](file:///c:/vs_code/wowml/wowml/_C/_bindings.py)
  - Tasks: Declare `argtypes`, `restype`, and error status checking for all C functions.
  - Acceptance Criteria: Raises Python exceptions on C errors with clear messages.

- [x] **8.2 Python Tensor Interop & NumPy Exchange**
  - Path: [`wowml/tensor.py`](file:///c:/vs_code/wowml/wowml/tensor.py)
  - Tasks: Wire `Tensor` creation, shape, strides, `numpy()`, and `tolist()` to the C kernel.
  - Acceptance Criteria: Zero-copy wrapping of NumPy arrays with `owns_data = 0`.

---

## Day 9: Autograd Engine & Core Differentiable Operators

- [x] **9.1 Dynamic Computational DAG & Topological Sort**
  - Path: [`wowml/autograd/engine.py`](file:///c:/vs_code/wowml/wowml/autograd/engine.py), [`wowml/autograd/function.py`](file:///c:/vs_code/wowml/wowml/autograd/function.py)
  - Tasks: Implement DFS topological sorting and reverse-mode `.backward()` gradient accumulation.
  - Acceptance Criteria: Correct accumulation on leaf `.grad` tensors when multiple paths converge.

- [x] **9.2 Implement 17 Differentiable Math Operators**
  - Path: [`wowml/autograd/grad_ops.py`](file:///c:/vs_code/wowml/wowml/autograd/grad_ops.py)
  - Tasks: Implement forward and backward for Add, Sub, Mul, Div, Neg, MatMul, Pow, Sqrt, Exp, Log, ReLU, Sigmoid, Tanh, Sum, Mean, Reshape, and Transpose.
  - Acceptance Criteria: Includes `_unbroadcast()` to sum out broadcasted dimensions.

---

## Day 10: Differentiable Linear Algebra & Advanced Autograd Features

- [ ] **10.1 Differentiable Solvers (`solve`, `cholesky`, `svd`)**
  - Path: [`wowml/autograd/linalg_ops.py`](file:///c:/vs_code/wowml/wowml/autograd/linalg_ops.py), [`wowml/linalg/`](file:///c:/vs_code/wowml/wowml/linalg/)
  - Tasks: `Solve` and `Cholesky` implemented; still need `SVD` autograd op and numerical gradient checks in `tests/test_linalg.py`.
  - Acceptance Criteria: Passes finite-difference numerical gradient checks in `tests/test_linalg.py`.

- [x] **10.2 Functional `autograd.grad` & Higher-Order Derivatives**
  - Path: [`wowml/autograd/engine.py`](file:///c:/vs_code/wowml/wowml/autograd/engine.py)
  - Tasks: Implement `grad(outputs, inputs, create_graph=True)`.
  - Acceptance Criteria: Allows taking the derivative of a derivative (e.g., Hessian-vector products).

- [x] **10.3 Graphviz DOT Graph Export**
  - Path: [`wowml/autograd/graph.py`](file:///c:/vs_code/wowml/wowml/autograd/graph.py)
  - Tasks: Implement `export_dot(root, filename)`.
  - Acceptance Criteria: Generates valid DOT file renderable by Graphviz.

---

## Day 11: Neural Network Architecture (`wowml.nn`)

- [x] **11.1 Module, Parameter & Serialization State**
  - Path: [`wowml/nn/module.py`](file:///c:/vs_code/wowml/wowml/nn/module.py), [`wowml/nn/parameter.py`](file:///c:/vs_code/wowml/wowml/nn/parameter.py)
  - Tasks: Implement `state_dict()` and `load_state_dict()` for parameter export and restore.
  - Acceptance Criteria: All child modules and parameters are discovered and serializable.

- [x] **11.2 Layers & Normalizations**
  - Path: [`wowml/nn/linear.py`](file:///c:/vs_code/wowml/wowml/nn/linear.py), [`wowml/nn/conv.py`](file:///c:/vs_code/wowml/wowml/nn/conv.py), [`wowml/nn/normalization.py`](file:///c:/vs_code/wowml/wowml/nn/normalization.py), [`wowml/nn/dropout.py`](file:///c:/vs_code/wowml/wowml/nn/dropout.py), [`wowml/nn/pooling.py`](file:///c:/vs_code/wowml/wowml/nn/pooling.py), [`wowml/nn/embedding.py`](file:///c:/vs_code/wowml/wowml/nn/embedding.py), [`wowml/nn/flatten.py`](file:///c:/vs_code/wowml/wowml/nn/flatten.py)
  - Tasks: Implement forward passes connecting through autograd functions.
  - Acceptance Criteria: Output dimensions and gradients match PyTorch reference layers.

- [x] **11.3 Activations & Losses**
  - Path: [`wowml/nn/activations.py`](file:///c:/vs_code/wowml/wowml/nn/activations.py), [`wowml/nn/loss.py`](file:///c:/vs_code/wowml/wowml/nn/loss.py)
  - Tasks: Implement all activations (including LeakyReLU, GELU, SiLU) and losses (MSELoss, CrossEntropyLoss, L1Loss, BCELoss, BCEWithLogitsLoss).
  - Acceptance Criteria: Forward and backward evaluations verify against PyTorch.

---

## Day 12: Optimizers & Learning Rate Schedulers (`wowml.optim`)

- [x] **12.1 First-Order Optimizers**
  - Path: [`wowml/optim/sgd.py`](file:///c:/vs_code/wowml/wowml/optim/sgd.py), [`wowml/optim/adam.py`](file:///c:/vs_code/wowml/wowml/optim/adam.py), [`wowml/optim/adamw.py`](file:///c:/vs_code/wowml/wowml/optim/adamw.py), [`wowml/optim/rmsprop.py`](file:///c:/vs_code/wowml/wowml/optim/rmsprop.py), [`wowml/optim/adagrad.py`](file:///c:/vs_code/wowml/wowml/optim/adagrad.py)
  - Tasks: Implement parameter update rules with momentum, weight decay, and buffer states.
  - Acceptance Criteria: Optimizers minimize quadratic loss in unit tests.

- [x] **12.2 Learning Rate Schedulers**
  - Path: [`wowml/optim/lr_scheduler.py`](file:///c:/vs_code/wowml/wowml/optim/lr_scheduler.py)
  - Tasks: Implement `StepLR` and `CosineAnnealingLR`.
  - Acceptance Criteria: Learning rates decay according to exact analytical schedules.

---

## Day 13: Data Pipeline & Custom Binary Serialization

- [x] **13.1 Datasets, Samplers, and DataLoader**
  - Path: [`wowml/data/`](file:///c:/vs_code/wowml/wowml/data/)
  - Tasks: Implement `Dataset`, `TensorDataset`, `SequentialSampler`, `RandomSampler`, and `DataLoader` (with `drop_last` and collation).
  - Acceptance Criteria: Batches iterate cleanly with matching batch sizes.

- [x] **13.2 Custom Binary Format without Pickle**
  - Path: [`wowml/utils/serialization.py`](file:///c:/vs_code/wowml/wowml/utils/serialization.py)
  - Tasks: Implement magic header `WOWML01`, schema metadata, and raw float buffer dumping/loading.
  - Acceptance Criteria: Secure, fast file round-trip saving and restoring models and tensors.

---

## Day 14: Pytest Suite, Benchmarks vs PyTorch/JAX & Release Verification

- [x] **14.1 Full Pytest Test Suite**
  - Path: [`tests/`](file:///c:/vs_code/wowml/tests/)
  - Tasks: Run `python -m pytest tests/ -v` covering tensors, numerical gradient checks, linalg, nn, optim, data, and serialization (`25/25` passed).
  - Acceptance Criteria: 100% test pass rate.

- [ ] **14.2 Performance Benchmarking Suite vs PyTorch & JAX**
  - Path: [`benchmarks/`](file:///c:/vs_code/wowml/benchmarks/)
  - Tasks: Benchmark harnesses implemented and `bench_matmul.py` / `bench_autograd.py` run; link OpenBLAS (`-DWOWML_USE_BLAS=ON`) to close gap with PyTorch/NumPy MKL (`49.64 GFLOPS` pure C11 vs `179.41 GFLOPS` PyTorch on `2048x2048`) and run `bench_linalg.py` / `bench_mlp.py`.
  - Acceptance Criteria: CPU throughput meets or exceeds PyTorch reference benchmarks on equivalent matrix sizes.

- [ ] **14.3 End-to-End Runnable Examples**
  - Path: [`examples/linear_regression.py`](file:///c:/vs_code/wowml/examples/linear_regression.py), [`examples/mnist.py`](file:///c:/vs_code/wowml/examples/mnist.py)
  - Tasks: Wire and verify training convergence on synthetic regression and MNIST handwritten digit classification.
  - Acceptance Criteria: Both scripts execute from terminal without error and achieve expected convergence.
