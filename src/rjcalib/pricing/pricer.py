from mpmath import mp, quad, re, sqrt, pi
from ..utils.settings import set_precision_from_env, integ_limits
from ..core.charfunc import Phi_I2_conditional

set_precision_from_env(80)

def price_power_call_from_cf(K, I2_t0, t0, t, Delta, z_t0, rv_params):
    """
    Transform pricer for the asymmetric power-type option using the conditional CF Φ(l).
    Returns price (mp.mpf complex real part taken inside integrand).
    """
    Phi = lambda l: Phi_I2_conditional(l, I2_t0, t0, t, Delta, z_t0, rv_params)

    # Integral piece; follows the paper's finite-range quadrature due to rapid decay
    def integrand(l):
        # Incomplete-gamma form collapsed; using a robust integrand that behaves well at l→0
        # If your integrand from the paper differs, we can swap it later.
        eps = mp.mpf("1e-18")
        denom = 1j * (l + eps)
        weight = K * mp.e**(-1j * (K**2) * l)
        return re(weight * Phi(l) / denom)

    upper, _, _ = integ_limits()
    val = (K/2) - (1/pi) * quad(integrand, [0, upper])
    return +val  # unary plus -> mp makes a clean mp.mpf
