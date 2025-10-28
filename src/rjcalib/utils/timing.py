import time, contextlib

@contextlib.contextmanager
def tic(name="block"):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[timing] {name}: {dt:.4f}s")
