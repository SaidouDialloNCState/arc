import sys, pathlib
p = pathlib.Path(__file__).parent / "src"
if str(p) not in sys.path:
    sys.path.insert(0, str(p))
