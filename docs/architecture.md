# WowML Architecture

## Overview

WowML is a three-layer system:

```
┌─────────────────────────────────────────────┐
│          Python API  (wowml/)               │
│  Tensor · nn · optim · data · linalg        │
├─────────────────────────────────────────────┤
│        Autograd Engine (wowml/autograd/)     │
│  Dynamic computation graph, reverse-mode AD  │
├─────────────────────────────────────────────┤
│        ctypes Bindings (wowml/_C/)           │
│  Loads wowml_c.{so,dll} at import time       │
├─────────────────────────────────────────────┤
│          C Kernel (csrc/)                    │
│  Tensor · ops · matmul · linalg · memory     │
└─────────────────────────────────────────────┘
```

## Data Flow

1. User creates a `Tensor` in Python.
2. Tensor data is allocated in C via `wowml_tensor_alloc()`.
3. Operations (add, matmul, etc.) call into C, return new Tensors.
4. If `requires_grad=True`, each op records a `Function` node in the autograd graph.
5. `tensor.backward()` traverses the graph in reverse topological order.
6. Optimizers read `.grad` on each Parameter and update `.data`.

## Key Design Decisions

- **ctypes over CPython extensions**: Simpler build, no Python.h dependency.
- **Dynamic graph**: Built on-the-fly (like PyTorch), not compiled.
- **Row-major storage**: C-contiguous layout for cache efficiency.
- **No GPU for v0.1**: CPU-only; GPU backend is a future milestone.
