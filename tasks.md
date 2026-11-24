# WowML — Implementation Tasks

> Master checklist of everything to implement and connect.
> Work through the phases in order — each phase builds on the previous one.

---

## Phase 1: C Kernel Foundation

The C layer is the performance backbone. Everything else calls into it.

### 1.1 Memory & Types (`csrc/src/memory.c`, `csrc/src/dtype.c`)
- [x] `wowml_alloc` / `wowml_calloc` / `wowml_free` — basic allocator wrapper
- [x] `wowml_dtype_size` / `wowml_dtype_name` — dtype helpers
- [x] `wowml_set_allocator` — pluggable allocator
- [ ] Add aligned allocation (`_aligned_malloc` / `posix_memalign`) for SIMD
- [ ] Add allocation tracking / debug leak checker

### 1.2 Tensor Core (`csrc/src/tensor.c`)
- [x] `wowml_tensor_alloc` — allocate data buffer with row-major strides
- [x] `wowml_tensor_free` — free tensor data
- [x] `wowml_tensor_zeros` — allocate + zero-fill
- [x] `wowml_tensor_print` — debug printing
- [ ] `wowml_tensor_ones` — allocate + fill with 1.0 (dtype-aware loop)
- [ ] `wowml_tensor_fill` — fill with arbitrary scalar value
- [ ] `wowml_tensor_clone` — deep copy (alloc + memcpy)
- [ ] `wowml_tensor_set_flat` / `wowml_tensor_get_flat` — flat-index element access with dtype dispatch
- [ ] `wowml_tensor_reshape` — validate numel match, update shape/strides in-place
- [ ] `wowml_tensor_transpose` — swap strides for 2-D; generalize for N-D
- [ ] Add `wowml_tensor_from_data` — wrap external buffer (non-owning)
- [ ] Add strided iteration helpers (for non-contiguous views)

### 1.3 Element-wise Ops (`csrc/src/ops.c`)
- [ ] Implement dtype dispatch macro (switch on `dtype`, cast `data` to typed pointer)
- [ ] `wowml_add` / `wowml_sub` / `wowml_mul` / `wowml_div` — element-wise binary ops
- [ ] `wowml_add_scalar` / `wowml_mul_scalar` — scalar broadcast ops
- [ ] `wowml_neg` — negate all elements
- [ ] `wowml_exp` / `wowml_log` / `wowml_sqrt` / `wowml_pow_op` — math unary ops
- [ ] `wowml_relu` — max(0, x) element-wise
- [ ] `wowml_sigmoid` — 1/(1+exp(-x))
- [ ] `wowml_tanh_op` — tanh(x)
- [ ] `wowml_sum` / `wowml_mean` / `wowml_max` / `wowml_min` — axis reductions
- [ ] `wowml_equal` — element-wise comparison
- [ ] **Broadcasting engine** — shape compatibility check + virtual stride expansion

### 1.4 Matrix Multiplication (`csrc/src/matmul.c`)
- [ ] `wowml_dot` — 1-D dot product
- [ ] `wowml_matvec` — matrix × vector
- [ ] `wowml_matmul` — 2-D matrix × matrix (naive triple loop first)
- [ ] `wowml_bmm` — batched matmul for 3-D tensors
- [ ] **Optimization pass**: tiled matmul with cache-friendly blocking
- [ ] **Optimization pass**: SIMD intrinsics (SSE/AVX) for inner loop

### 1.5 Linear Algebra (`csrc/src/linalg.c`)
- [ ] `wowml_linalg_trace` — sum of diagonal elements
- [ ] `wowml_linalg_norm` — L1, L2, Linf norms
- [ ] `wowml_linalg_lu` — Doolittle LU with partial pivoting
- [ ] `wowml_linalg_det` — determinant via LU
- [ ] `wowml_linalg_solve` — forward/backward substitution after LU
- [ ] `wowml_linalg_inv` — inverse via LU or Gauss-Jordan
- [ ] `wowml_linalg_cholesky` — Cholesky-Banachiewicz for SPD matrices
- [ ] `wowml_linalg_qr` — Householder QR decomposition
- [ ] `wowml_linalg_eigh` — Jacobi eigenvalue algorithm for symmetric matrices
- [ ] `wowml_linalg_svd` — Golub-Kahan bidiagonalization + QR iteration

### 1.6 Error Handling (`csrc/src/error.c`)
- [x] `wowml_status_string` — status code to string
- [x] `wowml_set_error_handler` — custom callback
- [ ] Add `WOWML_CHECK(expr, status)` macro for internal assertions
- [ ] Wire error handler into all functions (call handler on failure)

### 1.7 C Tests (`csrc/tests/`)
- [ ] `test_tensor.c` — alloc/free, zeros/ones/fill, clone, set/get, reshape, transpose
- [ ] `test_linalg.c` — test each decomposition against known matrices
- [ ] Add `test_ops.c` — element-wise ops, broadcasting, reductions
- [ ] Add `test_matmul.c` — dot, matvec, matmul, bmm correctness

---

## Phase 2: Python ↔ C Bindings

Bridge the C kernel to Python so every Python op runs C code.

### 2.1 ctypes Bindings (`wowml/_C/_bindings.py`)
- [ ] Define `ctypes.Structure` for `wowml_tensor_t` matching the C struct layout
- [ ] Declare argtypes/restype for every C function in `lib`
- [ ] Wrap `wowml_tensor_alloc`, `wowml_tensor_free`, `wowml_tensor_zeros`, `wowml_tensor_ones`
- [ ] Wrap all ops: `wowml_add`, `wowml_sub`, `wowml_mul`, `wowml_div`, etc.
- [ ] Wrap `wowml_matmul`, `wowml_dot`, `wowml_matvec`, `wowml_bmm`
- [ ] Wrap all linalg functions
- [ ] Add error-checking wrapper: check return status, raise Python exception on error
- [ ] Support data transfer: Python list/numpy → C buffer and back

### 2.2 Python Tensor backed by C (`wowml/tensor.py`)
- [ ] Store a `wowml_tensor_t` ctypes struct instead of Python lists
- [ ] Implement `Tensor.__init__` — accept list/nested-list, allocate C tensor, copy data in
- [ ] Implement `shape`, `ndim`, `numel` properties from the C struct
- [ ] Implement `Tensor.zeros()`, `Tensor.ones()`, `Tensor.randn()` factories via C calls
- [ ] Implement `to_list()` / `to_numpy()` — read data back from C buffer
- [ ] Implement `__repr__` with actual data display
- [ ] Implement `__del__` to call `wowml_tensor_free`

---

## Phase 3: Autograd Engine

Reverse-mode automatic differentiation on top of the C-backed tensors.

### 3.1 Computation Graph (`wowml/autograd/function.py`)
- [ ] Finalize `Function.apply()` — only wire graph when any input has `requires_grad`
- [ ] Handle mix of Tensor and scalar inputs gracefully
- [ ] Implement `no_grad()` context manager to disable graph recording

### 3.2 Engine (`wowml/autograd/engine.py`)
- [ ] Implement `_topological_sort` — DFS-based reverse topological ordering
- [ ] Implement `backward()`:
  - Seed gradient (ones_like if scalar output, or user-provided)
  - Walk topo order, call `_grad_fn.backward()`
  - Accumulate `.grad` on leaf tensors (sum if multiple paths)
- [ ] Handle in-place grad accumulation (+=) correctly
- [ ] Add cycle detection / error on non-DAG graphs

### 3.3 Gradient Ops (`wowml/autograd/grad_ops.py`)
Implement `forward()` and `backward()` for each:
- [ ] `Add` — forward: C `wowml_add`; backward: pass-through
- [ ] `Sub` — forward: C `wowml_sub`; backward: (grad, -grad)
- [ ] `Mul` — forward: C `wowml_mul`; backward: (grad*b, grad*a)
- [ ] `Div` — forward: C `wowml_div`; backward: (grad/b, -grad*a/b²)
- [ ] `Neg` — forward: C `wowml_neg`; backward: -grad
- [ ] `MatMul` — forward: C `wowml_matmul`; backward: (grad@B^T, A^T@grad)
- [ ] `Pow` — forward: C `wowml_pow_op`; backward: n*x^(n-1)*grad
- [ ] `Sqrt` — backward: grad / (2*sqrt(x))
- [ ] `Exp` — backward: exp(x) * grad
- [ ] `Log` — backward: grad / x
- [ ] `ReLU` — backward: grad * (x > 0)
- [ ] `Sigmoid` — backward: grad * σ(x) * (1 - σ(x))
- [ ] `Tanh` — backward: grad * (1 - tanh²(x))
- [ ] `Sum` — backward: broadcast grad to input shape
- [ ] `Mean` — backward: grad / numel, broadcast
- [ ] `Reshape` — backward: reshape grad to original shape
- [ ] `Transpose` — backward: transpose grad back

### 3.4 Wire Tensor operators to Autograd
- [ ] `Tensor.__add__` → `Add.apply(self, other)`
- [ ] `Tensor.__sub__` → `Sub.apply(self, other)`
- [ ] `Tensor.__mul__` → `Mul.apply(self, other)`
- [ ] `Tensor.__truediv__` → `Div.apply(self, other)`
- [ ] `Tensor.__matmul__` → `MatMul.apply(self, other)`
- [ ] `Tensor.__neg__` → `Neg.apply(self)`
- [ ] `Tensor.sum()` → `Sum.apply(self)`
- [ ] `Tensor.mean()` → `Mean.apply(self)`
- [ ] `Tensor.reshape()` → `Reshape.apply(self, shape)`
- [ ] `Tensor.backward()` → `autograd.engine.backward(self)`
- [ ] Handle `__radd__`, `__rmul__`, etc. for scalar-tensor ops
- [ ] `Tensor.detach()` — return copy with `requires_grad=False`, no `_grad_fn`

### 3.5 Graph Utilities (`wowml/autograd/graph.py`)
- [ ] `trace_graph` — walk graph, print node names and shapes
- [ ] `count_parameters` — count leaf tensors with `requires_grad`

---

## Phase 4: Linear Algebra Python Layer

### 4.1 Wrappers (`wowml/linalg/`)
- [ ] `decompositions.py` — call C kernel, return Python Tensors for LU, QR, SVD, Cholesky, Eigh
- [ ] `solve.py` — `solve(A, b)` and `inv(A)` via C kernel
- [ ] `norms.py` — `norm()`, `det()`, `trace()` via C kernel
- [ ] Add autograd support for differentiable linalg ops (solve, cholesky, svd) — stretch goal

---

## Phase 5: Neural Network Module

### 5.1 Parameter & Module (`wowml/nn/parameter.py`, `wowml/nn/module.py`)
- [ ] Verify `Parameter` auto-registers in `Module._parameters` via `__setattr__`
- [ ] Add `Module.state_dict()` / `Module.load_state_dict()`
- [ ] Add `Module.to(dtype)` for casting all parameters

### 5.2 Layers
- [ ] `Linear.forward` — `x @ W.T + b` using autograd MatMul + Add
- [ ] `Linear.__init__` — Kaiming/Xavier weight initialization via `Tensor.randn`
- [ ] `Conv2d.forward` — im2col + matmul convolution
- [ ] `Conv2d.__init__` — proper kernel weight initialization
- [ ] Add `BatchNorm1d` / `BatchNorm2d` — running mean/var, affine params
- [ ] Add `Dropout` — random zeroing during training
- [ ] Add `Flatten` layer
- [ ] Add `MaxPool2d` / `AvgPool2d`
- [ ] Add `Embedding` layer

### 5.3 Activations (`wowml/nn/activations.py`)
- [ ] `ReLU.forward` → `grad_ops.ReLU.apply(x)`
- [ ] `Sigmoid.forward` → `grad_ops.Sigmoid.apply(x)`
- [ ] `Tanh.forward` → `grad_ops.Tanh.apply(x)`
- [ ] `Softmax.forward` — exp(x - max) / sum(exp(x - max))
- [ ] Add `LeakyReLU`, `GELU`, `SiLU`

### 5.4 Loss Functions (`wowml/nn/loss.py`)
- [ ] `MSELoss.forward` — `((pred - target) ** 2).mean()`
- [ ] `CrossEntropyLoss.forward` — log-softmax + NLL
- [ ] Add `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`

### 5.5 Containers (`wowml/nn/container.py`)
- [ ] Verify `Sequential` correctly chains modules
- [ ] Verify `ModuleList` registers children for `parameters()`

---

## Phase 6: Optimizers

### 6.1 SGD (`wowml/optim/sgd.py`)
- [ ] Implement basic SGD: `p.data -= lr * p.grad`
- [ ] Add momentum: velocity buffer per parameter
- [ ] Add weight decay (L2 regularization)
- [ ] Add Nesterov momentum (stretch)

### 6.2 Adam (`wowml/optim/adam.py`)
- [ ] Implement Adam with bias correction
- [ ] First moment (m) and second moment (v) buffers
- [ ] `m_hat = m / (1 - β1^t)`, `v_hat = v / (1 - β2^t)`
- [ ] `p -= lr * m_hat / (√v_hat + ε)`
- [ ] Add weight decay (AdamW variant)

### 6.3 Additional Optimizers (stretch)
- [ ] Add `RMSprop`
- [ ] Add `Adagrad`
- [ ] Add learning rate schedulers (`StepLR`, `CosineAnnealingLR`)

---

## Phase 7: Data Pipeline

### 7.1 Dataset (`wowml/data/dataset.py`)
- [ ] Verify abstract base class contract (`__len__`, `__getitem__`)
- [ ] Add `TensorDataset(x, y)` — wraps two tensors as a dataset

### 7.2 DataLoader (`wowml/data/dataloader.py`)
- [ ] Implement `_collate` — stack list of samples into batched Tensors
- [ ] Add `drop_last` option
- [ ] Add multi-worker loading (stretch — use `multiprocessing`)
- [ ] Add `Sampler` base class and `RandomSampler`

---

## Phase 8: Testing & Validation

### 8.1 C Tests
- [ ] Comprehensive tensor tests (all creation, access, shape ops)
- [ ] Ops tests with known inputs/outputs
- [ ] Matmul correctness (compare with naive Python)
- [ ] Linalg tests: verify decompositions reconstruct the input
  - LU: `P @ L @ U ≈ A`
  - QR: `Q @ R ≈ A`, `Q^T @ Q ≈ I`
  - SVD: `U @ diag(S) @ V^T ≈ A`
  - Cholesky: `L @ L^T ≈ A`

### 8.2 Python Tests
- [ ] `test_tensor.py` — creation, arithmetic, shape ops end-to-end through C
- [ ] `test_autograd.py` — gradient correctness for every op
- [ ] Numerical gradient checking: `(f(x+h) - f(x-h)) / 2h ≈ autograd grad`
- [ ] `test_linalg.py` — decomposition results against numpy
- [ ] `test_nn.py` — Linear, Conv2d forward shapes; loss function values
- [ ] `test_optim.py` — verify SGD/Adam converge on simple quadratic loss

---

## Phase 9: Integration & Examples

### 9.1 End-to-end examples
- [ ] `examples/basic_tensor.py` — create tensors, do arithmetic, print results
- [ ] `examples/linear_regression.py` — full train loop: data → model → loss → backward → step
- [ ] `examples/mnist.py` — MLP on MNIST with DataLoader, train/eval loop, accuracy reporting

### 9.2 Build & CI
- [ ] Verify CMake + setup.py builds on Windows (MSVC), Linux (gcc), macOS (clang)
- [ ] Add GitHub Actions CI: build C → run C tests → pip install → run Python tests
- [ ] Generate test coverage reports

---

## Phase 10: Performance & Polish (Stretch Goals)

### 10.1 C Kernel Optimization
- [ ] Tiled/blocked matmul for cache efficiency
- [ ] SIMD vectorization (SSE4/AVX2) for element-wise ops and matmul inner loop
- [ ] OpenMP threading for large tensor ops
- [ ] Optional BLAS backend (link to OpenBLAS/MKL if available)

### 10.2 Python API Polish
- [ ] Pretty-print tensors (formatted matrix display, truncation for large tensors)
- [ ] `Tensor.to(dtype)` — dtype casting
- [ ] Slicing / indexing support (`tensor[0, :, 2:5]`)
- [ ] `Tensor.contiguous()` — compact non-contiguous views

### 10.3 Additional Features
- [ ] Serialization: custom binary format (replace pickle)
- [ ] GPU backend (CUDA/OpenCL) — future major milestone
- [ ] RNN / LSTM layers
- [ ] Attention / Transformer layers
- [ ] `autograd.grad()` — compute gradients without accumulating on `.grad`
- [ ] Higher-order gradients (grad of grad)
- [ ] Computation graph visualization (export to Graphviz DOT)

---

## Dependency Graph

Shows which phases must complete before others can begin:

```
Phase 1 (C Kernel)
  │
  ├──► Phase 2 (Bindings)
  │       │
  │       ├──► Phase 3 (Autograd)
  │       │       │
  │       │       ├──► Phase 5 (NN Module)
  │       │       │       │
  │       │       │       ├──► Phase 6 (Optimizers)
  │       │       │       │       │
  │       │       │       │       └──► Phase 9 (Examples)
  │       │       │       │
  │       │       │       └──► Phase 7 (Data Pipeline)
  │       │       │
  │       │       └──► Phase 8 (Testing) ◄── all phases
  │       │
  │       └──► Phase 4 (Linalg Python)
  │
  └──► Phase 10 (Optimization) ← can start any time after Phase 1
```
