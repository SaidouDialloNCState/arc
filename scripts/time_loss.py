import sys, pathlib
from mpmath import mp
sys.path.insert(0, str(pathlib.Path(".").resolve() / "src"))
from rjcalib.data.market import MarketSet
from rjcalib.calib.objective import loss_l2, vec_to_theta
from rjcalib.utils.timing import tic

ms = MarketSet.from_csv(".data/sample_vix.csv", I2_t0=0.1, z_t0=0.0)
theta = vec_to_theta([0.5, 1.0, 0.3, 1.2, 0.7])
with tic("loss_l2(sample)"):
    v = loss_l2(ms, theta)
print("loss:", float(v))
