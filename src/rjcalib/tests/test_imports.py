def test_imports():
    import importlib
    for m in ["rjcalib", "rjcalib.core.kernels", "rjcalib.core.charfunc",
              "rjcalib.pricing.pricer", "rjcalib.calib.objective",
              "rjcalib.calib.optimize", "rjcalib.data.market"]:
        assert importlib.import_module(m)
