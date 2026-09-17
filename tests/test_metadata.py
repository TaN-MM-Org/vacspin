"""Version and metadata consistency: __init__, pyproject and
CITATION.cff must agree, so a release cannot ship a torn identity."""
import pathlib
import re

import vacspin

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_versions_agree():
    v = vacspin.__version__
    py = (ROOT / "pyproject.toml").read_text()
    assert re.search(rf'^version = "{re.escape(v)}"$', py, re.M)
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf'^version: {re.escape(v)}$', cff, re.M)


def test_all_exports_exist():
    for name in vacspin.__all__:
        assert hasattr(vacspin, name), name
