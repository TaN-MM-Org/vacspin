"""The worked tutorials in examples/ run, print what their docstrings
and the README say they print, and reach the right answers: the
calibrated values lie within 3 error bars of the simulated truth, the
best window equals its closed form, and the predicted fidelity agrees
with the fidelity of the simulated center within its error bar. The
lifetime tutorial (0.8.0) lands within 3 error bars of the simulated
lifetimes, of the design's Purcell factor and coupling rate, and of the
fidelity of the simulated lifetimes."""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TUTORIALS = ("tutorial_readout_window.py", "tutorial_spin_to_readout.py",
             "tutorial_lifetime_to_readout.py")

# since 0.8.0 the source distribution also ships examples/; this guard
# remains for installs that copy the tests alone
pytestmark = pytest.mark.skipif(
    not all((EXAMPLES / name).exists() for name in TUTORIALS),
    reason="examples/ is not present (run from a repository clone)")


def _run(name):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    res = subprocess.run([sys.executable, str(EXAMPLES / name)],
                         capture_output=True, text=True, env=env,
                         cwd=str(ROOT), timeout=120)
    assert res.returncode == 0, res.stderr
    return res.stdout


def _documented(name):
    """The output block in the tutorial's docstring."""
    text = (EXAMPLES / name).read_text(encoding="utf-8")
    doc = text.split('"""')[1]
    block = doc.split("It printed, with vacspin")[1].split("\n", 1)[1]
    lines = [ln[4:] for ln in block.strip("\n").splitlines()]
    return "\n".join(lines) + "\n"


def _num(pattern, text):
    return [float(x) for x in re.search(pattern, text).groups()]


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_output_is_documented(name):
    """The README pastes the same output as the docstring, and the
    numbers printed now equal the documented ones (to the last printed
    digit, compared as numbers)."""
    doc = _documented(name)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert doc in readme
    out = _run(name)
    nums = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")
    got = [float(x) for x in nums.findall(out)]
    want = [float(x) for x in nums.findall(doc)]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g == pytest.approx(w, rel=1e-12, abs=1e-12)


def test_readout_tutorial_reaches_the_truth():
    out = _run("tutorial_readout_window.py")
    for name in ("eta", "lam", "noise_rate"):
        val, err, true = _num(
            name + r"\s+= ([\d.]+) \+/- ([\d.]+).*simulated ([\d.]+)", out)
        assert abs(val - true) < 3.0 * err
    (p,) = _num(r"p = ([\d.]+)", out)
    assert p > 0.01
    win, dwin, closed = _num(r"best window: ([\d.]+) \+/- ([\d.]+) us "
                             r"\(closed form for threshold 1: ([\d.]+)",
                             out)
    assert win == closed
    f, df = _num(r"fidelity there: ([\d.]+) \+/- ([\d.]+)", out)
    t_true, f_true = _num(r"simulated values: ([\d.]+) us, fidelity "
                          r"([\d.]+)", out)
    assert abs(win - t_true) < 3.0 * dwin
    assert abs(f - f_true) < 3.0 * df


def test_spin_tutorial_reaches_the_truth():
    out = _run("tutorial_spin_to_readout.py")
    for name in ("lam_g", "ups_g"):
        val, err, true = _num(
            name + r" = ([\d.]+) \+/- ([\d.]+) GHz.*simulated ([\d.]+)",
            out)
        assert abs(val - true) < 3.0 * err
    f, df = _num(r"fidelity ([\d.]+) \+/- ([\d.]+)", out)
    (f_true,) = _num(r"simulated center: fidelity ([\d.]+)", out)
    assert abs(f - f_true) < 3.0 * df
    assert f > 0.99


def test_lifetime_tutorial_reaches_the_truth():
    out = _run("tutorial_lifetime_to_readout.py")
    for label in ("detuned", "on resonance"):
        val, err, true, p = _num(
            label + r":\s+([\d.]+) \+/- ([\d.]+) ns \(simulated "
            r"([\d.]+)\), p = ([\d.]+)", out)
        assert abs(val - true) < 3.0 * err
        assert p > 0.01
    f_c, df_c, f_design = _num(r"F_C = ([\d.]+) \+/- ([\d.]+) \(design "
                               r"([\d.]+)\)", out)
    assert abs(f_c - f_design) < 3.0 * df_c
    g, dg, g_design = _num(r"g = ([\d.]+) \+/- ([\d.]+) GHz \(design "
                           r"([\d.]+)\)", out)
    assert abs(g - g_design) < 3.0 * dg
    f, df, e_b, e_d = _num(r"fidelity ([\d.]+) \+/- ([\d.]+): bright "
                           r"error ([\d.]+), dark error ([\d.]+)", out)
    (f_true,) = _num(r"simulated lifetimes: fidelity ([\d.]+)", out)
    assert abs(f - f_true) < 3.0 * df
    # the fidelity is the mean of the two success probabilities
    assert abs(f - (1.0 - 0.5 * (e_b + e_d))) < 2e-5
    f1, e_b1, e_d1 = _num(r"threshold 1 instead: fidelity ([\d.]+), "
                          r"bright error ([\d.]+), dark error ([\d.]+)",
                          out)
    assert abs(f1 - (1.0 - 0.5 * (e_b1 + e_d1))) < 2e-5
    assert f1 < f and e_b1 < e_b and e_d1 > e_d
