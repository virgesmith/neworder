import numpy as np
import pytest

import neworder as no

# stochastic process ids
MORTALITY = no.SplitMix64.hash64("mortality")
FERTILITY = no.SplitMix64.hash64("fertility")


@pytest.fixture
def rs() -> no.SplitMix64:
    return no.SplitMix64(no.MonteCarlo.deterministic_identical_stream)


def test_ids() -> None:
    assert MORTALITY != FERTILITY


def test_repr(rs: no.SplitMix64) -> None:
    assert "SplitMix64" in repr(rs)
    assert "counter" not in repr(rs)


def test_repr_counter() -> None:
    rs = no.SplitMix64(no.MonteCarlo.deterministic_identical_stream, use_counter=True)
    assert "counter" in repr(rs)


def test_reset_counter() -> None:
    rs = no.SplitMix64(no.MonteCarlo.deterministic_identical_stream, use_counter=True)
    person_ids = np.arange(10, dtype=np.int64)
    rs.uarray(person_ids, MORTALITY, 2025)
    rs.uarray(person_ids, MORTALITY, 2025)
    assert rs.counter() == 2
    rs.reset()
    assert rs.counter() == 0


def test_uarray_1d(rs: no.SplitMix64) -> None:
    person_ids = np.arange(100, dtype=np.int64)
    out = rs.uarray(person_ids, MORTALITY, 2025)
    assert out.shape == (100,)
    assert out.dtype == np.float64
    assert np.all((out >= 0.0) & (out < 1.0))


def test_uarray_2d(rs: no.SplitMix64) -> None:
    person_ids = np.arange(50, dtype=np.int64)
    times = np.array([2025, 2026, 2027], dtype=np.int64)
    out = rs.uarray(person_ids, FERTILITY, times)
    assert out.shape == (50, 3)
    assert out.dtype == np.float64
    assert np.all((out >= 0.0) & (out < 1.0))


def test_uarray_3d(rs: no.SplitMix64) -> None:
    persons = np.arange(10, dtype=np.int64)
    draws = np.arange(4, dtype=np.int64)
    times = np.array([2025, 2026], dtype=np.int64)
    out = rs.uarray(persons, MORTALITY, times, draws)
    assert out.shape == (10, 2, 4)


def test_uarray_scalar_only(rs: no.SplitMix64) -> None:
    # All-scalar args produce a 0-d array
    out = rs.uarray(42, MORTALITY, 2025)
    assert out.shape == ()
    assert 0.0 <= float(out) < 1.0


def test_uarray_reproducible(rs: no.SplitMix64) -> None:
    person_ids = np.arange(200, dtype=np.int64)
    out1 = rs.uarray(person_ids, MORTALITY, 2025)
    out2 = rs.uarray(person_ids, MORTALITY, 2025)
    np.testing.assert_array_equal(out1, out2)


def test_uarray_reset_reproducible() -> None:
    rs = no.SplitMix64(no.MonteCarlo.deterministic_identical_stream)
    person_ids = np.arange(50, dtype=np.int64)
    out1 = rs.uarray(person_ids, MORTALITY, 2025)
    rs.reset()
    out2 = rs.uarray(person_ids, MORTALITY, 2025)
    np.testing.assert_array_equal(out1, out2)


def test_uarray_person_independence(rs: no.SplitMix64) -> None:
    # Value for person i must not depend on which other persons are in the array
    full = rs.uarray(np.array([0, 1, 2, 3], dtype=np.int64), MORTALITY, 2025)
    single = rs.uarray(np.array([2], dtype=np.int64), MORTALITY, 2025)
    assert full[2] == pytest.approx(single[0])


def test_uarray_process_independence(rs: no.SplitMix64) -> None:
    person_ids = np.arange(50, dtype=np.int64)
    mort = rs.uarray(person_ids, MORTALITY, 2025)
    fert = rs.uarray(person_ids, FERTILITY, 2025)
    assert not np.allclose(mort, fert)


def test_uarray_time_independence(rs: no.SplitMix64) -> None:
    person_ids = np.arange(50, dtype=np.int64)
    t2025 = rs.uarray(person_ids, MORTALITY, 2025)
    t2026 = rs.uarray(person_ids, MORTALITY, 2026)
    assert not np.allclose(t2025, t2026)


def test_uarray_uniform_distribution(rs: no.SplitMix64) -> None:
    # KS test: large sample should look uniform
    from scipy.stats import kstest

    person_ids = np.arange(10_000, dtype=np.int64)
    out = rs.uarray(person_ids, MORTALITY, 2025)
    _stat, p = kstest(out, "uniform")
    assert p > 0.01


def test_uarray_bad_arg(rs: no.SplitMix64) -> None:
    with pytest.raises(TypeError):
        rs.uarray("not_an_int_or_array")  # ty: ignore[invalid-argument-type]


def test_uarray_2d_input_rejected(rs: no.SplitMix64) -> None:
    with pytest.raises(TypeError):
        rs.uarray(np.zeros((3, 3), dtype=np.int64))


def test_uarray_no_args(rs: no.SplitMix64) -> None:
    with pytest.raises(ValueError):
        rs.uarray()


def test_uarray_scalar_vs_single_element_array_shape(rs: no.SplitMix64) -> None:
    # Scalar arg: no output dimension -> shape (3, 4)
    a = rs.uarray(np.array([1, 2, 3]), 4, np.array([5, 6, 7, 8]))
    assert a.shape == (3, 4)

    # Single-element array: size-1 dimension preserved -> shape (3, 1, 4)
    b = rs.uarray(np.array([1, 2, 3]), np.array([4]), np.array([5, 6, 7, 8]))
    assert b.shape == (3, 1, 4)


def test_uarray_scalar_vs_single_element_array_values(rs: no.SplitMix64) -> None:
    # A scalar arg is premixed into the salt; a single-element array arg is a data
    # dimension hashed on top of the salt. They have the same shape footprint but
    # a different position in the hash chain, so their values intentionally differ.
    a = rs.uarray(np.array([1, 2, 3]), 4, np.array([5, 6, 7, 8]))
    b = rs.uarray(np.array([1, 2, 3]), np.array([4]), np.array([5, 6, 7, 8]))
    assert not np.allclose(a, b[:, 0, :])


# --- use_counter tests ---


@pytest.fixture
def rs_counter() -> no.SplitMix64:
    return no.SplitMix64(no.MonteCarlo.deterministic_identical_stream, use_counter=True)


def test_counter_initial(rs_counter: no.SplitMix64) -> None:
    assert rs_counter.counter() == 0


def test_counter_increments(rs_counter: no.SplitMix64) -> None:
    person_ids = np.arange(10, dtype=np.int64)
    for expected in range(5):
        assert rs_counter.counter() == expected
        rs_counter.uarray(person_ids, MORTALITY, 2025)
    assert rs_counter.counter() == 5


def test_counter_reset(rs_counter: no.SplitMix64) -> None:
    person_ids = np.arange(10, dtype=np.int64)
    rs_counter.uarray(person_ids, MORTALITY, 2025)
    rs_counter.uarray(person_ids, MORTALITY, 2025)
    assert rs_counter.counter() == 2
    rs_counter.reset()
    assert rs_counter.counter() == 0


def test_counter_unique_streams(rs_counter: no.SplitMix64) -> None:
    # Same args on consecutive calls must produce different results
    person_ids = np.arange(100, dtype=np.int64)
    out1 = rs_counter.uarray(person_ids, MORTALITY, 2025)
    out2 = rs_counter.uarray(person_ids, MORTALITY, 2025)
    assert not np.allclose(out1, out2)


def test_counter_reset_reproducible(rs_counter: no.SplitMix64) -> None:
    # After reset the sequence restarts identically
    person_ids = np.arange(50, dtype=np.int64)
    out1 = rs_counter.uarray(person_ids, MORTALITY, 2025)
    out2 = rs_counter.uarray(person_ids, FERTILITY, 2026)
    rs_counter.reset()
    rep1 = rs_counter.uarray(person_ids, MORTALITY, 2025)
    rep2 = rs_counter.uarray(person_ids, FERTILITY, 2026)
    np.testing.assert_array_equal(out1, rep1)
    np.testing.assert_array_equal(out2, rep2)


def test_no_counter_unchanged(rs: no.SplitMix64) -> None:
    # Without use_counter, counter stays at 0 and repeated calls return the same values
    person_ids = np.arange(50, dtype=np.int64)
    out1 = rs.uarray(person_ids, MORTALITY, 2025)
    out2 = rs.uarray(person_ids, MORTALITY, 2025)
    assert rs.counter() == 0
    np.testing.assert_array_equal(out1, out2)
