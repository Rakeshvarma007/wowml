# WowML — Official User & Reference Documentation

**Version:** 0.1.0  
**Target Runtime:** Python 3.10+ · C11 + OpenMP (+ optional OpenBLAS / LAPACKE)

---

## 1. What Is WowML and How Does It Work?

**WowML** is a CPU-first Python machine learning library backed by a compiled C11 shared library (`wowml_c.dll` / `libwowml_c.so` / `libwowml_c.dylib`). It separates high-level model definition from low-level numerical execution across four layers:

1. **C11 Compute Kernel (`csrc/`)**:
   - Stores tensor payloads in **64-byte SIMD-aligned** row-major buffers (`wowml_tensor_t`).
   - Executes element-wise arithmetic (with N-D broadcasting), unary math, reductions, `im2col`/`col2im`, tiled + OpenMP matrix multiplication (with optional OpenBLAS `cblas_sgemm`/`cblas_dgemm`), and linear algebra decompositions.
2. **Zero-Copy `ctypes` Bridge (`wowml._C`)**:
   - Maps `wowml_tensor_t` directly to Python via `ctypes.Structure` (`CWowmlTensor`).
   - Supports zero-copy wrapping of contiguous NumPy arrays (`owns_data = 0`) so data is shared in-place without redundant memory copies.
3. **Dynamic Reverse-Mode Autograd (`wowml.autograd`)**:
   - Records operations on-the-fly into a Directed Acyclic Graph (DAG) whenever `is_grad_enabled()` is `True` and at least one input tensor has `requires_grad=True`.
   - Traverses the DAG in reverse topological order during `.backward()` or `autograd.grad()`.
4. **High-Level ML Modules (`wowml.nn`, `wowml.optim`, `wowml.linalg`, `wowml.data`, `wowml.utils`)**:
   - Provides PyTorch-style `Module`, `Parameter`, layers, losses, optimizers, learning-rate schedulers, datasets/dataloaders, and a secure `pickle`-free binary serialization format (`WOWML01`).

---

## 2. Installation & Building the C Kernel

Before importing and running operations in `wowml`, the C shared library must be compiled:

```bash
# 1. Configure CMake (Release mode, OpenMP, optional OpenBLAS)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
# (On Windows with MinGW-W64, add: -G "MinGW Makefiles")

# 2. Build the shared library and C tests
cmake --build build --config Release -j 8

# 3. Verify C kernel
cd build && ctest --output-on-failure -C Release && cd ..

# 4. Install Python package in editable mode
pip install -e ".[dev]"
```

> **Note:** On Windows, `wowml._C._bindings` automatically registers `C:/mingw64/bin` and `C:/msys64/ucrt64/bin` (if present) via `os.add_dll_directory` so `wowml_c.dll` and its OpenMP runtime (`libgomp-1.dll`) resolve automatically.

---

## 3. Core API Usage Guide

### 3.1 `wowml.Tensor`

```python
import numpy as np
import wowml
from wowml import Tensor

# Creation
a = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32", requires_grad=True)
z = Tensor.zeros(2, 3, dtype="float32")
o = Tensor.ones(2, 3, dtype="float64")
r = Tensor.randn(4, 4, dtype="float32")

# Zero-copy NumPy interop (shares memory; NumPy array must stay alive)
arr = np.ones((2, 2), dtype=np.float32)
t_np = Tensor.from_numpy(arr, requires_grad=False)

# Operations
c = a + o[:2, :2] if False else a + Tensor.ones(2, 2)
d = a @ Tensor.ones(2, 2)
s = d.sum()

# Export back to NumPy / Python list
np_view = d.numpy()   # Zero-copy NumPy view
py_list = d.tolist()
```

**Supported `dtype` strings:**
- `"float32"` (aliases: `"f32"`, `"float"`)
- `"float64"` (aliases: `"f64"`, `"double"`)
- `"int32"` (alias: `"i32"`)
- `"int64"` (alias: `"i64"`)

---

### 3.2 Automatic Differentiation (`wowml.autograd`)

```python
from wowml import Tensor
from wowml.autograd import grad, no_grad

x = Tensor([2.0, -3.0], requires_grad=True)
y = (x * x).sum()

# 1. Standard backward pass (accumulates into x.grad)
y.backward()
print(x.grad.numpy())  # [4.0, -6.0]

# Clear gradients before the next iteration
x.zero_grad()

# 2. Functional gradient API (does not mutate .grad; supports higher-order via create_graph=True)
z = (x * x * x).sum()
(dz_dx,) = grad(z, x, create_graph=False)

# 3. Disable gradient tracking for inference / evaluation
with no_grad():
    out = x * 2.0
    assert not out.requires_grad
```

---

### 3.3 Linear Algebra (`wowml.linalg`)

```python
import wowml
from wowml import Tensor

A = Tensor([[4.0, 2.0], [2.0, 3.0]], dtype="float64")
b = Tensor([6.0, 5.0], dtype="float64")

x = wowml.linalg.solve(A, b)          # Solve Ax = b
inv_A = wowml.linalg.inv(A)           # Matrix inverse
L = wowml.linalg.cholesky(A)          # Lower triangular L where A = L @ L^T
Q, R = wowml.linalg.qr(A)             # Reduced QR: A = Q @ R
U, S, Vt = wowml.linalg.svd(A)        # SVD: A = U @ diag(S) @ V^T
w, V = wowml.linalg.eigh(A)           # Symmetric eigendecomposition
lu, piv = wowml.linalg.lu(A)          # LU with partial pivoting

d = wowml.linalg.det(A)               # Determinant (float)
tr = wowml.linalg.trace(A)            # Trace (float)
n2 = wowml.linalg.norm(A, ord=2)      # Norm (ord=1, 2, or -1 for inf)
```

---

### 3.4 Neural Networks & Optimizers (`wowml.nn`, `wowml.optim`)

```python
import numpy as np
import wowml.nn as nn
import wowml.optim as optim
from wowml import Tensor

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.StepLR(optimizer, step_size=10, gamma=0.5)

x = Tensor(np.random.randn(8, 4).astype(np.float32))
y = Tensor(np.random.randn(8, 2).astype(np.float32))

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

**Available `nn` components:**
- Layers: `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`, `Dropout`, `MaxPool2d`, `AvgPool2d`, `Embedding`, `Flatten`
- Containers: `Sequential`, `ModuleList`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `GELU`, `SiLU`
- Losses: `MSELoss`, `L1Loss`, `CrossEntropyLoss`, `BCELoss`, `BCEWithLogitsLoss`

**Available `optim` components:**
- Optimizers: `SGD` (with `momentum`, `weight_decay`, `nesterov`), `Adam`, `AdamW`, `RMSprop`, `Adagrad`
- Schedulers: `StepLR`, `CosineAnnealingLR`

---

### 3.5 Data Pipeline & Binary Serialization (`wowml.data`, `wowml.utils`)

```python
import wowml
from wowml.data import Dataset, DataLoader

# Custom dataset
class MyDataset(Dataset):
    def __len__(self): return 100
    def __getitem__(self, idx): return idx, idx * 2

loader = DataLoader(MyDataset(), batch_size=16, shuffle=True, drop_last=False)
for batch in loader:
    pass

# Save and load state_dict or Tensor (secure WOWML01 binary format, no pickle)
wowml.utils.save(model.state_dict(), "checkpoint.wml")
state = wowml.utils.load("checkpoint.wml")
model.load_state_dict(state, strict=True)
```

---

## 4. What Should NOT Be Given (Input Restrictions)

To prevent runtime failures, silent corruption, or undefined behavior, **never** pass the following:

1. **Ragged Nested Python Lists**:
   - *Do not pass:* `Tensor([[1.0, 2.0], [3.0]])` → raises `ValueError`.
2. **Tensors Exceeding 8 Dimensions**:
   - The C kernel enforces `WOWML_MAX_DIMS = 8`. Passing `ndim <= 0` or `ndim > 8` raises `ValueError`.
3. **Unsupported NumPy or Python Dtypes**:
   - *Do not pass:* `float16`, `bfloat16`, `uint8`, `complex64`, `bool`, or string arrays to `Tensor.from_numpy()`. Only `float32`, `float64`, `int32`, and `int64` are supported.
4. **Mismatched Dtypes Across Binary Ops or MatMul**:
   - *Do not mix* `float32` and `float64` in `a + b` or `a @ b` (no implicit type promotion). Both operands must share the exact same `dtype`.
5. **Non-Positive-Definite or Non-Square Matrices to Restricted Solvers**:
   - `cholesky(A)` requires a **symmetric positive-definite (SPD)** 2D square matrix.
   - `solve(A, b)`, `inv(A)`, `det(A)`, `trace(A)`, `eigh(A)` require **2D square** matrices (`ndim == 2` and `shape[0] == shape[1]`).
6. **Non-Tensor Objects in `wowml.utils.save()`**:
   - `save()` only accepts a `Tensor` or a `dict` mapping string keys to `Tensor` values (e.g., `model.state_dict()`). Do not pass arbitrary Python classes, functions, or `nn.Module` instances directly (save `model.state_dict()` instead).
7. **Mutating a NumPy Array Wrapped with `Tensor.from_numpy()` While Autograd Tracks It**:
   - Because `from_numpy()` is zero-copy (`owns_data = 0`), mutating the underlying NumPy buffer in-place invalidates saved tensors needed by `.backward()`.
8. **Floating-Point Targets for `CrossEntropyLoss`**:
   - `CrossEntropyLoss(logits, targets)` expects `logits` of shape `(B, C)` and integer class-index `targets` of shape `(B,)` in range `[0, C - 1]`, not one-hot float vectors.

---

## 5. Error Reference: What Could Result in an Error?

All C kernel functions return a `wowml_status_t` code (`csrc/include/wowml/error.h`), which `wowml._C._bindings.check_status()` translates into Python exceptions:

| C Status Code | Python Exception | Common Causes |
| :--- | :--- | :--- |
| `WOWML_ERR_NULL_PTR` (1) | `RuntimeError` | Unallocated `Tensor` (`_c_tensor is None`), `None` passed where a C pointer is required. |
| `WOWML_ERR_SHAPE_MISMATCH` (2) | `ValueError` | Incompatible shapes in `a + b` (not broadcastable), `a @ b` where `a.shape[1] != b.shape[0]`, or `reshape` with mismatched `numel`. |
| `WOWML_ERR_OUT_OF_MEMORY` (3) | `MemoryError` | System RAM exhausted during `wowml_aligned_alloc` or temporary workspace allocation. |
| `WOWML_ERR_INVALID_DTYPE` (4) | `ValueError` | Mixed dtypes in an op, integer tensor passed to float-only math (`exp`, `log`, `matmul`, `linalg`), or unsupported dtype string. |
| `WOWML_ERR_SINGULAR_MATRIX` (5) | `ValueError` | Calling `solve` or `inv` on a singular/rank-deficient matrix, or `cholesky` on a non-positive-definite matrix. |
| `WOWML_ERR_INVALID_ARGUMENT` (6) | `ValueError` | `ndim > 8` or `<= 0`, negative shape dimensions, out-of-bounds `axis`, `T` on non-2D tensor, or `step <= 0` in slicing. |
| `WOWML_ERR_NOT_IMPLEMENTED` (7) | `NotImplementedError` | Calling an op or feature not yet implemented in the current build (e.g., general `Tensor.__getitem__` multi-axis slicing). |

### Additional Python-Level Errors

- **`RuntimeError: The wowml_c shared library is not loaded.`**:
  - Raised when `wowml_c.dll` / `libwowml_c.so` has not been built or cannot be found in `wowml/_C/`, `build/`, or `build/Release/`. Fix by building with CMake.
- **`RuntimeError: grad can be implicitly created only for scalar outputs.`**:
  - Raised when calling `tensor.backward()` on a non-scalar tensor (`tensor.numel != 1`) without passing an explicit `gradient` tensor of matching shape.
- **`KeyError` in `model.load_state_dict(state, strict=True)`**:
  - Raised when keys in `state` do not match the model's parameter names.
- **`ValueError: Invalid WowML binary file (bad magic header).`**:
  - Raised by `wowml.utils.load()` if the file is corrupted, empty, or created by `pickle` / another library instead of `wowml.utils.save()`.
