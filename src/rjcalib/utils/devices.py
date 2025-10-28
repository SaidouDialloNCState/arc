def get_xp():
    """
    Returns a NumPy-like module: CuPy if available, else NumPy.
    Usage:
        xp = get_xp()
        a = xp.array([1,2,3])
    """
    try:
        import cupy as cp
        _ = cp.zeros(1)  # sanity
        return cp
    except Exception:
        import numpy as np
        return np
