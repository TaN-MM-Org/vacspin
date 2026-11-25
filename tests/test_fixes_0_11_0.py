"""0.11.0 fixes: field inputs that gave silent nonsense or unclear
errors."""
import numpy as np
import pytest

import vacspin as vs


def test_field_on_circle_refuses_bad_inputs():
    """0.10.0 returned [nan, 0, nan] for a NaN angle, and for a negative
    size the field pointing the other way. Both are refused now, as in
    `field_from_axis`; a size of 0 is still allowed."""
    for angle, size in ((np.nan, 0.1), (np.inf, 0.1), (10.0, -0.1),
                        (10.0, np.nan), (10.0, np.inf)):
        with pytest.raises(ValueError, match="b_mag_t"):
            vs.field_on_circle(angle, size)
    assert np.array_equal(vs.field_on_circle(30.0, 0.0), np.zeros(3))
    b = vs.field_on_circle(147.0, 0.125)
    assert np.linalg.norm(b) == pytest.approx(0.125, rel=1e-15)


def test_field_must_hold_three_numbers():
    """0.10.0 stopped with NumPy's matmul error for two or four
    components and with a TypeError for a 3 x 3 array. Now each is a
    ValueError that names the expected vector; a (3, 1) column still
    works and gives the same result as (3,)."""
    p = vs.snv_rosenthal2023()
    b = vs.field_on_circle(147.0, 0.125)
    for bad in ([0.1, 0.1], [0.1, 0.1, 0.1, 0.1], np.full((3, 3), 0.1),
                "abc"):
        with pytest.raises(ValueError, match="three numbers"):
            vs.qubit_frequency(p, bad)
        with pytest.raises(ValueError, match="three numbers"):
            vs.rabi_rate(p, b, bad)
    assert vs.qubit_frequency(p, b.reshape(3, 1)) == vs.qubit_frequency(p, b)
