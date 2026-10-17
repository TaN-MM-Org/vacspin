"""The worked tutorials in examples/ run, print what their docstrings
and the README say they print, and reach the right answers: the
calibrated values lie within 3 error bars of the simulated truth, the
best window equals its closed form, and the predicted fidelity agrees
with the fidelity of the simulated center within its error bar."""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TUTORIALS = ("tutorial_readout_window.py", "tutorial_spin_to_readout.py")

# the source distribution ships the tests but not examples/
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
