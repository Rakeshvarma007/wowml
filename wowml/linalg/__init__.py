"""
linalg — Linear algebra operations (Python wrappers around C kernel).
"""
from wowml.linalg.decompositions import lu, qr, svd, cholesky, eigh  # noqa: F401
from wowml.linalg.solve import solve, inv                             # noqa: F401
from wowml.linalg.norms import norm, det, trace                       # noqa: F401
