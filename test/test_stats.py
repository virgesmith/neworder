import warnings

import numpy as np
import pytest

import neworder as no


# TODO remove along with the deprecated neworder.stats submodule
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


def test_stats_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="neworder.stats is deprecated") as record:
        _ = no.stats
    assert record[0].filename == __file__

    with pytest.warns(DeprecationWarning, match="neworder.stats is deprecated") as record:
        from neworder import stats  # noqa: F401
    assert record[0].filename == __file__


def test_no_warning_without_stats() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = no.df
        _ = no.time
        with pytest.raises(AttributeError):
            _ = no.not_an_attribute  # ty: ignore[unresolved-attribute]
