import json, subprocess, sys, pathlib, os

def test_cli_runs_and_outputs(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[3]  # repo root
    csv = root / ".data" / "sample_vix.csv"
    out = tmp_path / "res.json"
    dump = tmp_path / "modeled.csv"
    env = os.environ.copy()
    env["RJ_USE_FAST_HDELTA"] = "1"
    cmd = [sys.executable, "-m", "rjcalib.calib.run",
           "--csv", str(csv), "--I2_t0", "0.1", "--z_t0", "0.0",
           "--maxiter_global", "3", "--maxiter_local", "5",
           "--use_fast_H", "--parallel", "2",
           "--out", str(out), "--dump_prices", str(dump)]
    r = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert out.exists(), "no JSON output"
    d = json.loads(out.read_text())
    assert "x" in d and "fun" in d and isinstance(d["x"], list)
    assert dump.exists(), "no dump csv"
    # basic sanity: modeled csv should contain 'model' column
    txt = dump.read_text()
    assert "model" in txt
