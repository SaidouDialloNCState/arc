from mpmath import mp
from rjcalib.data.market import MarketSet
from rjcalib.calib.objective import loss_l2, vec_to_theta

def test_smoke_pricer_and_loss():
    ms = MarketSet.from_csv(".data/sample_vix.csv", I2_t0=0.1, z_t0=0.0)
    theta = vec_to_theta([0.5, 1.0, 0.3, 1.2, 0.7])
    val = loss_l2(ms, theta)
    assert mp.isfinite(val)
