import warnings

import numpy as np
import pytest
from scipy import special

import neworder as no


# TODO remove along with the deprecated neworder.stats functions
def test_logistic_logit() -> None:
    n = 100  # wont work if odd!

    x = np.linspace(-10.0, 10.0, n + 1)
    with pytest.warns(DeprecationWarning):
        y = no.stats.logistic(x)  # ty: ignore[deprecated]
    assert np.all(y >= -1)
    assert np.all(y <= 1)
    assert y[n // 2] == 0.5

    assert np.all(np.fabs(y + y[::-1] - 1.0) < 1e-15)

    with pytest.warns(DeprecationWarning):
        x2 = no.stats.logit(y)  # ty: ignore[deprecated]

    assert np.all(np.fabs(x2 - x) < 2e-12)


def test_scipy_equivalents() -> None:
    x = np.linspace(-10.0, 10.0, 11)
    with pytest.warns(DeprecationWarning):
        assert np.allclose(no.stats.logistic(x, 1.5, 0.5), special.expit(0.5 * (x - 1.5)))  # ty: ignore[deprecated]
    p = np.linspace(0.05, 0.95, 10)
    with pytest.warns(DeprecationWarning):
        assert np.allclose(no.stats.logit(p), special.logit(p))  # ty: ignore[deprecated]


def test_deprecation_warnings() -> None:
    x = np.array([0.25, 0.5])
    with pytest.warns(DeprecationWarning, match=r"neworder\.stats\.logistic is deprecated") as record:
        no.stats.logistic(x)  # ty: ignore[deprecated]
    assert record[0].filename == __file__

    with pytest.warns(DeprecationWarning, match=r"neworder\.stats\.logit is deprecated") as record:
        no.stats.logit(x)  # ty: ignore[deprecated]
    assert record[0].filename == __file__

    # accessing the submodule or the functions themselves doesn't warn, only calling them
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from neworder import stats

        assert stats.logistic.__name__ == "logistic"  # ty: ignore[deprecated]
        assert "logistic function" in (stats.logistic.__doc__ or "")  # ty: ignore[deprecated]
