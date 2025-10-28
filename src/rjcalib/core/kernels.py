from mpmath import mp, quad, exp, power, gamma

mp.dps = 80  # high precision for stability

def h_typeIII(t_minus_s, kappa, d):
    """
    Piecewise kernel h(t-s) for d in (0.5, 1). Matches the Type III form.
    """
    # thresholds
    thresh = (1 - d) / kappa
    if t_minus_s < thresh:
        return (power(t_minus_s, d - 1)
                - power((1 - d) / kappa, d - 1) / gamma(d)
                - power(kappa, 1 - d) / (power(1 - d, 2 - d) * gamma(d - 1)))
    else:
        return -(exp(1) ** (kappa * (1 - d))) * exp(-kappa * (t_minus_s)) / (power(1 - d, 2 - d) * gamma(d - 1))

def H_delta(t_minus_s, kappa, d, Delta):
    """
    Delta-forward integrated kernel H_Δ(t,s) = (1/Δ) ∫_0^Δ h(t+u - s) du .
    Uses closed forms; falls back to numeric integration for safety.
    """
    thresh = (1 - d) / kappa
    x = t_minus_s
    try:
        if x + Delta < thresh:
            return (power(x + Delta, d) - power(x, d)) / (Delta * gamma(d + 1))
        elif x < thresh <= x + Delta:
            # Mixed region: first part below thresh, remainder above
            left = (power((1 - d) / kappa, d) - power(x, d)) / (Delta * gamma(d + 1))
            right = (exp(-kappa * (x + Delta)) - exp(-kappa * thresh)) \
                    / (Delta * (power(1 - d, 2 - d) * gamma(d - 1))) * (exp(1) ** (kappa * (1 - d)))
            return left + right
        else:
            return - (exp(1) ** (kappa * (1 - d))) * (exp(-kappa * x) - exp(-kappa * (x + Delta))) \
                   / (Delta * (power(1 - d, 2 - d) * gamma(d - 1)))
    except Exception:
        f = lambda u: h_typeIII(x + u, kappa, d)
        return quad(f, [0, Delta]) / Delta
