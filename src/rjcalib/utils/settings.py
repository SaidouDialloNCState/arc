import os
from mpmath import mp

def set_precision_from_env(default_dps=80):
    dps = int(os.environ.get("RJ_MPMATH_DPS", default_dps))
    mp.dps = dps   # <-- correct attribute
    return dps

def integ_limits():
    ell_upper = float(os.environ.get("RJ_ELL_UPPER", "50"))
    x_left    = float(os.environ.get("RJ_X_LEFT",  "-30"))
    x_right   = float(os.environ.get("RJ_X_RIGHT", "30"))
    return ell_upper, x_left, x_right
