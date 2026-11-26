# WowML

[![WowML CI](https://github.com/your-org/wowml/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/wowml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**WowML** is a high-performance machine learning library featuring a hybrid **C11 + OpenBLAS/LAPACKE** compute kernel, dynamic reverse-mode **autograd**, a complete **linear algebra** suite, modular **neural network** primitives, and **optimizers** designed to achieve CPU parity with PyTorch and JAX.

---

## Key Architectural Highlights

- **Hybrid C11 + OpenBLAS/LAPACKE Kernel (`csrc/`)**:
  - Handcrafted C11 reference implementations for multi-dimensional strided tensor arithmetic, zero-copy slicing, and N-D broadcasting.
  - Optional OpenBLAS / LAPACKE fast paths (`-DWOWML_USE_BLAS=ON`) and multi-threaded parallel execution via **OpenMP**.
  - Compiler optimization flags (`-O3 -march=native` or `/O2 /arch:AVX2`).
  - SIMD-aligned allocations (64-byte alignment) and an arena allocator for scratch workspace buffers.
  - C-level `im2col`/`col2im` and fused operators (`fused_linear_bias_relu`, `fused_cross_entropy`).

- **Zero-Copy ctypes Interoperability (`wowml._C`)**:
  - Transparent data exchange with NumPy arrays using explicit memory ownership semantics (`owns_data`).

- **Dynamic Reverse-Mode Autograd (`wowml.autograd`)**:
  - Topological graph execution, functional `autograd.grad()`, higher-order gradients, `no_grad()` contexts, and Graphviz DOT graph export.
  - Differentiable linear algebra solvers (`solve`, `cholesky`, `svd`).

- **Neural Network Primitives (`wowml.nn`)**:
  - `Module`, `Parameter`, `state_dict` / `load_state_dict`.
  - Layers: `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`, `Dropout`, `Flatten`, `MaxPool2d`, `AvgPool2d`, `Embedding`.
  - Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `GELU`, `SiLU`.
  - Losses: `MSELoss`, `CrossEntropyLoss`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`.

- **Optimizers & Schedulers (`wowml.optim`)**:
  - `SGD` (with momentum, weight decay, and Nesterov), `Adam`, `AdamW`, `RMSprop`, `Adagrad`.
  - `StepLR`, `CosineAnnealingLR`.

- **Data Pipeline & Secure Serialization (`wowml.data`, `wowml.utils`)**:
  - `Dataset`, `TensorDataset`, `Sampler`, `RandomSampler`, and `DataLoader` (with multi-worker and `drop_last` support).
  - Secure custom binary model serialization without `pickle`.

---

## Project Structure

```text
wowml/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix
├── benchmarks/                    # CPU benchmark suite vs PyTorch & JAX
├── csrc/                          # C11 + BLAS/LAPACKE compute kernel
│   ├── include/wowml/             # Public C headers
│   │   ├── wowml.h                # Umbrella header
│   │   ├── error.h                # Status codes & assertion macros
│   │   ├── dtype.h                # Data types (float32, float64, int32, int64)
│   │   ├── memory.h               # SIMD aligned & Arena memory allocators
│   │   ├── tensor.h               # wowml_tensor_t, slicing, views, broadcasting
│   │   ├── ops.h                  # Element-wise math, im2col, fused kernels
│   │   ├── matmul.h               # Tiled matmul, bmm, BLAS dispatch
│   │   └── linalg.h               # LU, QR, SVD, Cholesky, Eigh, solve, inv, det
│   ├── src/                       # C implementation files
│   └── tests/                     # C unit test harnesses (ctest)
├── docs/                          # Architecture guides & diagrams
├── examples/                      # End-to-end runnable scripts
├── tests/                         # Pytest test suites
├── wowml/                         # Python package
│   ├── _C/                        # ctypes bridge to wowml_c shared library
│   ├── autograd/                  # Dynamic reverse-mode autodiff engine
│   ├── linalg/                    # Linear algebra wrappers
│   ├── nn/                        # Layers, activations, losses, containers
│   ├── optim/                     # Optimizers & learning rate schedulers
│   ├── data/                      # Dataset, Samplers, and DataLoader
│   └── utils/                     # Binary model serialization
├── CMakeLists.txt                 # CMake configuration
├── CONTRIBUTING.md                # Development guidelines & standards
├── devhelp.text                   # Technical onboarding & architecture guide
├── pyproject.toml                 # Package metadata
├── setup.py                       # CMake-driven setuptools build script
└── tasks.md                       # 14-day full-scope engineering roadmap
```

---

## Build & Quick Start

### 1. Build the C Kernel with CMake
```bash
# Configure with BLAS (optional) & OpenMP enabled
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# On Windows with MinGW-W64:
# cmake -B build -S . -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Compile the shared library and C tests
cmake --build build --config Release -j 8

# Run C kernel unit tests
cd build && ctest --output-on-failure -C Release && cd ..
```

### 2. Install WowML in Editable Mode
```bash
pip install -e ".[dev]"
```

### 3. Run the Python Test Suite
```bash
python -m pytest tests/ -v
```

### 4. Run Benchmarks against PyTorch & JAX
```bash
python -m benchmarks.bench_matmul
python -m benchmarks.bench_autograd
python -m benchmarks.bench_linalg
python -m benchmarks.bench_mlp
```

---

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.
