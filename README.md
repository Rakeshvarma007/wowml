# WowML(Work in progress)

A Python machine learning library powered by a custom C kernel, with built-in linear algebra and automatic differentiation.

## Features

- **C Kernel** — High-performance tensor operations written in C
- **Linear Algebra** — Matrix decompositions, solvers, norms (LU, QR, SVD, Cholesky, Eigen)
- **Autograd** — Reverse-mode automatic differentiation with dynamic computation graphs
- **Neural Networks** — Modular `nn.Module` API (Linear, Conv2d, activations, losses)
- **Optimizers** — SGD, Adam, and extensible optimizer base
- **Data Pipeline** — Dataset and DataLoader abstractions

## Quick Start

```bash
# Build the C kernel
mkdir build && cd build
cmake .. && make

# Install the Python package
pip install -e ".[dev]"
```

## License

MIT
