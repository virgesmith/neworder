from math import factorial

import numpy as np
import pytest

import neworder as no


def test_nondeterministic_reset() -> None:
    mc = no.MonteCarlo(no.MonteCarlo.nondeterministic_stream)
    seeds = {mc.seed()}
    for _ in range(5):
        mc.reset()
        seeds.add(mc.seed())
    assert len(seeds) == 6


def test_deterministic_reset() -> None:
    mc = no.MonteCarlo(no.MonteCarlo.deterministic_identical_stream)
    seeds = {mc.seed()}
    for _ in range(5):
        mc.reset()
        seeds.add(mc.seed())
    assert len(seeds) == 1


def test_mc_property(mc: no.MonteCarlo) -> None:
    mc.ustream(1)
    mc.reset()


def test_mc(mc: no.MonteCarlo) -> None:
    x = mc.ustream(1)
    mc.reset()
    assert x == mc.ustream(1)


def test_default_seeder() -> None:
    assert no.MonteCarlo().raw() == no.MonteCarlo(no.MonteCarlo.deterministic_independent_stream).raw()
    assert no.MonteCarlo().raw() != no.MonteCarlo(lambda: 42).raw()


def test_seeder_errors() -> None:
    assert no.MonteCarlo(lambda: np.int32(7)).seed() == 7  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="seeder's result must be an int32"):
        no.MonteCarlo(lambda: 1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="seeder's result must be an int32"):
        no.MonteCarlo(lambda: 2**31)
    with pytest.raises(ZeroDivisionError):
        no.MonteCarlo(lambda: 1 // 0)

    seeds = iter([1, "x"])
    mc = no.MonteCarlo(lambda: next(seeds))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="seeder's result must be an int32"):
        mc.reset()


def test_seeders() -> None:
    # serial tests
    # determinisitc seeders always return the same value
    assert no.MonteCarlo.deterministic_identical_stream() == no.MonteCarlo.deterministic_identical_stream()
    assert no.MonteCarlo.deterministic_independent_stream() == no.MonteCarlo.deterministic_independent_stream()
    # nondeterministic seeders don't
    assert no.MonteCarlo.nondeterministic_stream() != no.MonteCarlo.nondeterministic_stream()

    if not no.mpi.COMM:
        return

    # parallel tests
    # all seeds equal
    seeds = no.mpi.COMM.gather(no.MonteCarlo.deterministic_identical_stream(), 0)
    if no.mpi.RANK == 0:
        assert seeds
        assert len(seeds) == no.mpi.SIZE
        assert len(set(seeds)) == 1

    # all seeds different but reproducible
    seeds = no.mpi.COMM.gather(no.MonteCarlo.deterministic_independent_stream(), 0)
    if no.mpi.RANK == 0:
        assert seeds
        assert len(seeds) == no.mpi.SIZE
        assert len(set(seeds)) == len(seeds)
    seeds2 = no.mpi.COMM.gather(no.MonteCarlo.deterministic_independent_stream(), 0)
    if no.mpi.RANK == 0:
        assert seeds == seeds2

    # all seeds different and not reproducible
    seeds = no.mpi.COMM.gather(no.MonteCarlo.nondeterministic_stream(), 0)
    if no.mpi.RANK == 0:
        assert seeds
        assert len(seeds) == no.mpi.SIZE
        assert len(set(seeds)) == len(seeds)
    # TODO need higher time resolution on seeder
    seeds2 = no.mpi.COMM.gather(no.MonteCarlo.nondeterministic_stream(), 0)
    if no.mpi.RANK == 0:
        assert seeds != seeds2

    # test custom seeder
    def seeder() -> int:
        return no.mpi.RANK + 1

    mc = no.MonteCarlo(seeder)
    assert mc.seed() == no.mpi.RANK + 1


def test_sample(mc: no.MonteCarlo) -> None:
    with pytest.raises(ValueError):
        mc.sample(100, np.array([0.9]))
    with pytest.raises(ValueError):
        mc.sample(100, np.array([-0.1, 1.1]))
    with pytest.raises(ValueError):
        mc.sample(100, np.array([0.5, 1.5]))
    with pytest.raises(ValueError):
        mc.sample(100, [])
    assert np.all(mc.sample(100, np.array([1.0, 0.0, 0.0, 0.0])) == 0)
    assert np.all(mc.sample(100, np.array([0.0, 1.0, 0.0, 0.0])) == 1)
    assert np.all(mc.sample(100, np.array([0.0, 0.0, 0.0, 1.0])) == 3)


def test_hazard(mc: no.MonteCarlo) -> None:
    assert np.all(mc.hazard(0.0, 10) == 0.0)
    assert np.all(mc.hazard(1.0, 10) == 1.0)

    with pytest.raises(ValueError):
        mc.hazard(-0.1, 10)
    with pytest.raises(ValueError):
        mc.hazard(1.1, 10)
    with pytest.raises(ValueError):
        mc.hazard(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        mc.hazard(np.array([0.1, 1.2]))
    with pytest.raises(ValueError):
        mc.hazard(np.nan, 1)
    with pytest.raises(ValueError):
        mc.hazard(np.array([0.1, np.nan]))


def test_stopping(mc: no.MonteCarlo) -> None:
    assert np.all(mc.stopping(0.0, 10) == no.time.FAR_FUTURE)

    with pytest.raises(ValueError):
        mc.stopping(-0.1, 10)
    with pytest.raises(ValueError):
        mc.stopping(1.1, 10)
    with pytest.raises(ValueError):
        mc.stopping(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        mc.stopping(np.array([0.1, 1.2]))
    with pytest.raises(ValueError):
        mc.stopping(np.nan, 1)
    with pytest.raises(ValueError):
        mc.stopping(np.array([0.1, np.nan]))


def test_arrivals_validation(mc: no.MonteCarlo) -> None:
    assert np.all(no.time.isnever(mc.first_arrival([0.0, 0.0], 1.0, 10)))
    with pytest.raises(ValueError):
        mc.first_arrival(np.array([-1.0, 0.0]), 1.0, 10)
    with pytest.raises(ValueError):
        mc.first_arrival([1.0, np.nan], 1.0, 10)

    assert np.all(no.time.isnever(mc.next_arrival(np.zeros(10), [0.0, 0.0], 1.0)))
    with pytest.raises(ValueError):
        mc.next_arrival(np.zeros(10), [-1.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        mc.next_arrival(np.zeros(10), [np.nan, np.nan], 1.0)

    with pytest.raises(ValueError):
        mc.arrivals([-1.0, 0.0], 1.0, 10, 0.0)
    with pytest.raises(ValueError):
        mc.arrivals([1.0, 1.0], 1.0, 10, 0.0)
    with pytest.raises(ValueError):
        mc.arrivals([np.nan, np.nan], 1.0, 10, 0.0)

    with pytest.raises(ValueError):
        mc.first_arrival([], 1.0, 10)
    with pytest.raises(ValueError):
        mc.next_arrival(np.zeros(10), [], 1.0)
    with pytest.raises(ValueError):
        mc.arrivals([], 1.0, 10, 0.0)

    assert mc.arrivals([1.0, 0.0], 1.0, 0, 0.0).shape == (0, 0)


def test_mc_counts(mc: no.MonteCarlo) -> None:
    assert mc.seed() == 19937

    def poisson_pdf(x: range, lambda_: float) -> np.ndarray:
        y = np.exp(-lambda_)
        return np.array([lambda_**k * y / factorial(k) for k in x])

    tests = [
        (1.0, 1.0, 10000),
        (3.0, 0.5, 10000),
        (0.2, 2.0, 10000),
        (10.0, 1.0, 1000),
        (3.0, 1.0, 100000),
    ]

    for lam, dt, n in tests:
        c = mc.counts([lam] * n, dt)
        x = range(0, max(c))
        # convert to counts
        c1 = [(c == k).sum() / n for k in x]
        p = poisson_pdf(x, lam * dt)

        for i in x:
            assert np.fabs(c1[i] - p[i]) < 1.0 / np.sqrt(n)


def test_mc_serial(mc: no.MonteCarlo) -> None:
    if no.mpi.SIZE != 1:
        return

    assert mc.seed() == 19937

    mc.reset()
    assert mc.raw() == 6231104047474287856
    assert mc.raw() == 14999272868227999252
    mc.reset()
    assert mc.raw() == 6231104047474287856
    assert mc.raw() == 14999272868227999252

    mc.reset()
    s = mc.state()
    a = mc.ustream(5)
    assert s != mc.state()
    assert abs(a[0] - 0.33778882725164294) < 1e-8
    assert abs(a[1] - 0.04767065867781639) < 1e-8
    assert abs(a[2] - 0.8131122114136815) < 1e-8
    assert abs(a[3] - 0.24954832065850496) < 1e-8
    assert abs(a[4] - 0.3385562978219241) < 1e-8

    mc.reset()
    assert s == mc.state()
    h = mc.hazard(0.5, 1000000)
    assert np.sum(h) == 500151

    n = 10000
    # 10% constant hazard for 10 time units, followed by zero
    dt = 1.0
    p = np.full(11, 0.1)
    p[-1] = 0
    a = mc.first_arrival(p, dt, n)
    assert np.nanmin(a) > 0.0
    assert np.nanmax(a) < 10.0
    no.log(f"{np.nanmin(a)} - {np.nanmax(a)}")

    # now set a to all 8.0
    a = np.full(n, 8.0)
    # next arrivals (absolute) only in range 8-10, if they happen
    b = mc.next_arrival(a, p, dt)
    assert np.nanmin(b) > 8.0
    assert np.nanmax(b) < 10.0

    # next arrivals with gap dt (absolute) only in range 9-10, if they happen
    b = mc.next_arrival(a, p, dt, False, dt)
    assert np.nanmin(b) > 9.0
    assert np.nanmax(b) < 10.0

    # next arrivals (relative) only in range 8-18, if they happen
    b = mc.next_arrival(a, p, dt, True)
    assert np.nanmin(b) > 8.0
    assert np.nanmax(b) < 18.0

    # next arrivals with gap dt (relative) only in range 9-19, if they happen
    b = mc.next_arrival(a, p, dt, True, dt)
    assert np.nanmin(b) > 9.0
    assert np.nanmax(b) < 19.0

    # now set a back to random arrivals
    a = mc.first_arrival(p, dt, n)
    # next arrivals (absolute) only in range (min(a), 10), if they happen
    b = mc.next_arrival(a, p, dt)
    assert np.nanmin(b) > np.nanmin(a)
    assert np.nanmax(b) < 10.0

    # next arrivals with gap dt (absolute) only in range (min(a)+dt, 10), if they happen
    b = mc.next_arrival(a, p, dt, False, dt)
    assert np.nanmin(b) > np.nanmin(a) + dt
    assert np.nanmax(b) < 10.0

    # next arrivals (relative) only in range (min(a), max(a)+10), if they happen
    b = mc.next_arrival(a, p, dt, True)
    assert np.nanmin(b) > np.nanmin(a)
    assert np.nanmax(b) < np.nanmax(a) + 10.0

    # next arrivals with gap dt (relative) only in range (min(a)+dt, max(a)+dt+10), if they happen
    b = mc.next_arrival(a, p, dt, True, dt)
    assert np.nanmin(b) > np.nanmin(a) + dt
    assert np.nanmax(b) < np.nanmax(a) + dt + 10.0

    mc.reset()
    a = mc.first_arrival(np.array([0.1, 0.2, 0.3]), 1.0, 6, 0.0)
    assert len(a) == 6
    # only works for single-process
    assert a[0] == 3.6177811673165667
    assert a[1] == 0.6896205251312125
    assert a[2] == 3.610216282947799
    assert a[3] == 7.883336832344425
    assert a[4] == 6.461894711350323
    assert a[5] == 2.8566436418145944

    mc.reset()
    a = mc.arrivals([1.0, 2.0, 3.0, 0.0], 1.0, 1, 0.0)
    assert np.allclose(
        a[0],
        [
            0.361778116731657,
            0.430740169244778,
            1.580095480774,
            2.226284951909032,
            2.511949316090492,
            2.809348320658414,
            2.929632529913839,
        ],
    )
    mc.reset()
    # now with a mim separation of 1.0
    a = mc.arrivals([1.0, 2.0, 3.0, 0.0], 1.0, 1, 1.0)
    assert np.allclose(a[0], [0.361778116731657, 1.430740169244778])
    mc.reset()

    # Exp.value = p +/- 1/sqrt(N)
    h = mc.hazard(0.2, 10000)
    assert isinstance(h, np.ndarray)
    assert len(h) == 10000
    assert abs(np.mean(h) - 0.2) < 0.01

    hv = mc.hazard(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert isinstance(hv, np.ndarray)
    assert len(hv) == 5

    # Exp.value = 1/p +/- 1/sqrt(N)
    st = mc.stopping(0.1, 10000)
    assert isinstance(st, np.ndarray)
    assert len(st) == 10000
    assert abs(np.mean(st) / 10 - 1.0) < 0.03

    sv = mc.stopping(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert isinstance(sv, np.ndarray)
    assert len(sv) == 5

    # Non-homogeneous Poisson process (time-dependent hazard)
    nhpp = mc.first_arrival(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 1.0, 10, 0.0)
    assert isinstance(nhpp, np.ndarray)
    assert len(nhpp) == 10


def test_mc_parallel(mc: no.MonteCarlo, indep_mc: no.MonteCarlo) -> None:
    if no.mpi.SIZE == 1:
        return
    assert no.mpi.COMM is not None

    # test model has identical streams
    mc.reset()
    assert mc.seed() == 19937

    a = mc.ustream(5)
    all_a = no.mpi.COMM.gather(a, root=0)
    all_states = no.mpi.COMM.gather(mc.state(), root=0)

    if no.mpi.RANK == 0:
        assert all_a and all_states
        for r in range(0, no.mpi.SIZE):
            assert np.all(all_states[0] == all_states[r])
            assert np.all(a - all_a[r] == 0.0)

    # test model_i has independent streams
    mc = indep_mc
    mc.reset()
    assert mc.seed() == 19937 + no.mpi.RANK

    a = mc.ustream(5)
    all_a = no.mpi.COMM.gather(a, root=0)
    all_states = no.mpi.COMM.gather(mc.state(), root=0)

    # check all other streams different
    if no.mpi.RANK == 0:
        assert all_a and all_states
        for r in range(1, no.mpi.SIZE):
            assert not np.all(a - all_a[r] == 0.0)


def test_bitgen(mc: no.MonteCarlo) -> None:
    mc2 = no.MonteCarlo(no.MonteCarlo.deterministic_identical_stream)
    gen = no.as_np(mc)

    n = gen.bit_generator.random_raw()
    assert n == mc2.raw()
    assert (gen.uniform(size=100) == mc2.ustream(100)).all()

    # check the np gen gets the reset
    mc.reset()
    assert n == gen.bit_generator.random_raw()

    mc.reset()
    mc_different_seed = no.MonteCarlo(lambda: 1234)
    gen2 = no.as_np(mc_different_seed)
    assert gen2.bit_generator.random_raw() != mc.raw()
    assert (gen2.uniform(size=100) != mc.ustream(100)).all()


def test_strided_arrays(mc: no.MonteCarlo) -> None:
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert (mc.hazard(p[::2]) == 0.0).all()
    assert (mc.hazard(p[1::2]) == 1.0).all()
    assert (mc.hazard(p[::-1]) == [1.0, 0.0, 1.0, 0.0]).all()
