# Contributing to WowML

Thank you for your interest in contributing to **WowML**! This guide outlines our engineering standards, memory ownership principles, architectural conventions, and development workflows.

---

## Core Engineering Principles

1. **Zero Cuts, Complete Scope**: WowML is built to provide a complete, robust ML platform. Every module, layer, and kernel must have clean documentation, boundary checks, and full unit test coverage.
2. **C11 Standard**: All code in `csrc/` must adhere strictly to C11 standards. Avoid compiler-specific extensions unless guarded with feature-detection macros (`#ifdef _MSC_VER`, `#ifdef __GNUC__`).
3. **Dual Compute Backend**: Every compute-intensive operation (`matmul`, `bmm`, `lu`, `qr`, `svd`, `cholesky`, `solve`) provides:
   - A pure C11 reference implementation for portability and educational clarity.
   - An optional OpenBLAS / LAPACKE fast path enabled with `-DWOWML_USE_BLAS=ON` to match or beat PyTorch and JAX on CPU.
4. **Clean Stubs During Staging**: When scaffolding components, never write partial fake logic. Use clean stubs returning `WOWML_ERR_NOT_IMPLEMENTED` in C or raising `NotImplementedError` in Python.

---

## Memory Ownership & Striding Conventions

### Memory Ownership (`owns_data`)
- When allocating a fresh tensor via `wowml_tensor_alloc`, `owns_data` is set to `1`.
- When wrapping external buffers (e.g. from NumPy via ctypes) or creating zero-copy views (slicing, transposition, reshape), `owns_data` is set to `0`.
- Calling `wowml_tensor_free` on a tensor where `owns_data == 0` must NOT free the memory buffer, preventing double-free panics.
- Always use `wowml_aligned_alloc` / `wowml_aligned_free` for 64-byte boundary alignment required by AVX-512 and AVX2 vector instructions.

### Striding & Contiguity
- Default layout is N-D row-major:
  $$\text{stride}[d-1] = 1, \quad \text{stride}[i] = \text{stride}[i+1] \times \text{shape}[i+1]$$
- Slicing and transposition create views with modified strides and offsets without copying underlying data.
- If an operation requires a contiguous memory layout (e.g., BLAS or SIMD vectorization), call `wowml_tensor_contiguous` first.

---

## Adding a New Feature: End-to-End Workflow

When implementing a new operation or layer, follow this step-by-step pipeline:

```text
[1. C Header]       csrc/include/wowml/ops.h       -> Declare function signature
[2. C Kernel]       csrc/src/ops.c                 -> Implement C11 reference + BLAS hook
[3. C Unit Test]    csrc/tests/test_ops.c          -> Verify in C via ctest
[4. ctypes Bridge]  wowml/_C/_bindings.py          -> Expose function via ctypes
[5. Autograd Node]  wowml/autograd/grad_ops.py     -> Implement Function.forward & backward
[6. Tensor Hook]    wowml/tensor.py                -> Bind magic method or member method
[7. Pytest Test]    tests/test_autograd.py         -> Verify with numerical gradient check
[8. Benchmark]      benchmarks/                    -> Benchmark latency vs PyTorch/JAX
```

---

## Development Workflow & Commands

### Prerequisites
- CMake 3.20+
- C Compiler: MSVC (Visual Studio 2022 / Build Tools), GCC 10+, or Clang 12+
- Python 3.10+
- Optional: OpenBLAS / LAPACKE development libraries

### Building & Testing
```bash
# 1. Configure CMake
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# 2. Build C shared library and test targets
cmake --build build --config Release

# 3. Run C test suite
cd build && ctest --output-on-failure -C Release && cd ..

# 4. Install Python package in editable mode
pip install -e ".[dev]"

# 5. Run Python test suite
pytest tests/ -v
```

---

## Pull Request Checklist

Before submitting a pull request, ensure:
- [ ] Code builds without warnings under `-Wall -Wextra -Wpedantic` or MSVC `/W4`.
- [ ] No memory leaks (verify using AddressSanitizer or Valgrind on Linux/macOS).
- [ ] All unit tests pass (`ctest` and `pytest`).
- [ ] Analytical gradients pass finite-difference verification (`tests/test_autograd.py`).
- [ ] Documentation and type annotations are updated.
