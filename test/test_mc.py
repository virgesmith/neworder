import numpy as np
import neworder as no
import platform

from utils import assert_throws


def test_mc_property():
  model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  #model.mc = model.mc not allowed

  model.mc.ustream(1)
  model.mc.reset()


def test_mc():

  model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  x = model.mc.ustream(1)
  model.mc.reset()
  assert x == model.mc.ustream(1)

  if no.mpi.size() == 1:
    _test_mc_serial(model)
  else:
    model_i = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_independent_stream)
    _test_mc_parallel(model, model_i)

def test_seeders():

  # serial tests
  # determinisitc seeders always return the same value
  assert no.MonteCarlo.deterministic_identical_stream(no.mpi.rank()) == no.MonteCarlo.deterministic_identical_stream(no.mpi.rank())
  assert no.MonteCarlo.deterministic_independent_stream(no.mpi.rank()) == no.MonteCarlo.deterministic_independent_stream(no.mpi.rank())
  # nondeterministic seeders don't
  assert no.MonteCarlo.nondeterministic_stream(no.mpi.rank()) != no.MonteCarlo.nondeterministic_stream(no.mpi.rank())

  try:
    import mpi4py.MPI as mpi
    comm = mpi.COMM_WORLD
  except Exception:
    return

  # parallel tests

  # all seeds equal
  seeds = comm.gather(no.MonteCarlo.deterministic_identical_stream(no.mpi.rank()), 0)
  if no.mpi.rank() == 0:
    assert len(seeds) == no.mpi.size()
    assert len(set(seeds)) == 1

  # all seeds different but reproducible
  seeds = comm.gather(no.MonteCarlo.deterministic_independent_stream(no.mpi.rank()), 0)
  if no.mpi.rank() == 0:
    assert len(seeds) == no.mpi.size()
    assert len(set(seeds)) == len(seeds)
  seeds2 = comm.gather(no.MonteCarlo.deterministic_independent_stream(no.mpi.rank()), 0)
  if no.mpi.rank() == 0:
    assert seeds == seeds2

  # all seeds different and not reproducible
  seeds = comm.gather(no.MonteCarlo.nondeterministic_stream(no.mpi.rank()), 0)
  if no.mpi.rank() == 0:
    assert len(seeds) == no.mpi.size()
    assert len(set(seeds)) == len(seeds)
  # TODO need higher time resolution on seeder
  seeds2 = comm.gather(no.MonteCarlo.nondeterministic_stream(no.mpi.rank()), 0)
  if no.mpi.rank() == 0:
    assert seeds != seeds2

  # test custom seeder
  seeder = lambda r: r + 1
  m = no.Model(no.NoTimeline(), seeder)
  assert m.mc.seed() == no.mpi.rank() + 1


def test_sample():
  m = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  assert_throws(ValueError, m.mc.sample, 100, [0.9])
  assert_throws(ValueError, m.mc.sample, 100, [-0.1, 1.1])
  assert np.all(m.mc.sample(100, [1.0, 0.0, 0.0, 0.0]) == 0)
  assert np.all(m.mc.sample(100, [0.0, 1.0, 0.0, 0.0]) == 1)
  assert np.all(m.mc.sample(100, [0.0, 0.0, 0.0, 1.0]) == 3)

def test_hazard():
  m = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  assert np.all(m.mc.hazard(0.0,10) == 0.0)
  assert np.all(m.mc.hazard(1.0,10) == 1.0)

  assert_throws(ValueError, m.mc.hazard, -0.1, 10)
  assert_throws(ValueError, m.mc.hazard, 1.1, 10)
  assert_throws(ValueError, m.mc.hazard, [-0.1, 0.5])
  assert_throws(ValueError, m.mc.hazard, [0.1, 1.2])
  assert_throws(ValueError, m.mc.hazard, np.nan, 1)
  assert_throws(ValueError, m.mc.hazard, [0.1, np.nan])

def test_stopping():
  m = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  assert np.all(m.mc.stopping(0.0,10) == no.time.far_future())

  assert_throws(ValueError, m.mc.stopping, -0.1, 10)
  assert_throws(ValueError, m.mc.stopping,  1.1, 10)
  assert_throws(ValueError, m.mc.stopping, [-0.1, 0.5])
  assert_throws(ValueError, m.mc.stopping, [0.1, 1.2])
  assert_throws(ValueError, m.mc.stopping, np.nan, 1)
  assert_throws(ValueError, m.mc.stopping, [0.1, np.nan])

def test_arrivals_validation():
  m = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
  assert np.all(no.time.isnever(m.mc.first_arrival([0.0,0.0], 1.0, 10)))
  assert_throws(ValueError, m.mc.first_arrival, np.array([-1.0,0.0]), 1.0, 10)
  assert_throws(ValueError, m.mc.first_arrival, [1.0,np.nan], 1.0, 10)

  assert np.all(no.time.isnever(m.mc.next_arrival(np.zeros(10), [0.0,0.0], 1.0)))
  assert_throws(ValueError, m.mc.next_arrival, np.zeros(10), [-1.0,0.0], 1.0)
  assert_throws(ValueError, m.mc.next_arrival, np.zeros(10), [np.nan,np.nan], 1.0)

  assert_throws(ValueError, m.mc.arrivals, [-1.0,0.0], 1.0, 10, 0.0)
  assert_throws(ValueError, m.mc.arrivals, [1.0,1.0], 1.0, 10, 0.0)
  assert_throws(ValueError, m.mc.arrivals, [np.nan,np.nan], 1.0, 10, 0.0)

def test_mc_counts():
  m = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
  mc = m.mc
  assert mc.seed() == 19937

  def poisson_pdf(x, l):
    y = np.exp(-l)
    return np.array([l**k * y / np.math.factorial(k) for k in x])


  tests = [(1.0, 1.0, 10000), (3.0, 0.5, 10000), (0.2, 2.0, 10000), (10.0, 1.0, 1000), (3.0, 1.0, 100000)]

  for lam, dt, n in tests:

    c = mc.counts([lam] * n, dt)
    x = range(0,max(c))
    # convert to counts
    c = [(c == k).sum() / n for k in x]
    p = poisson_pdf(x, lam*dt)

    for i in x:
      assert np.fabs(c[i] - p[i]) < 1.0 / np.sqrt(n)

def _test_mc_serial(model):
  mc = model.mc
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
  if platform.system() != "Darwin":
    # mc.state() just returns 0 on OSX due to an apparent bug in MT19937 that intermittently segfaults
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
  no.log("%f - %f" % (np.nanmin(a), np.nanmax(a)))

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
  assert np.allclose(a[0], [0.361778116731657, 0.430740169244778, 1.580095480774, 2.226284951909032, 2.511949316090492, 2.809348320658414, 2.929632529913839])
  mc.reset()
  # now with a mim separation of 1.0
  a = mc.arrivals([1.0, 2.0, 3.0, 0.0], 1.0, 1, 1.0)
  assert np.allclose(a[0], [0.361778116731657, 1.430740169244778])
  mc.reset()

  # Exp.value = p +/- 1/sqrt(N)
  h = model.mc.hazard(0.2, 10000)
  assert isinstance(h, np.ndarray)
  assert len(h) == 10000
  assert abs(np.mean(h) - 0.2) < 0.01

  hv = model.mc.hazard(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  assert isinstance(hv, np.ndarray)
  assert len(hv) == 5

  # Exp.value = 1/p +/- 1/sqrt(N)
  s = model.mc.stopping(0.1, 10000)
  assert isinstance(s, np.ndarray)
  assert len(s) == 10000
  assert abs(np.mean(s)/10 - 1.0) < 0.03

  sv = model.mc.stopping(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
  assert isinstance(sv, np.ndarray)
  assert len(sv) == 5

  # Non-homogeneous Poisson process (time-dependent hazard)
  nhpp = model.mc.first_arrival(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 1.0, 10, 0.0)
  assert isinstance(nhpp, np.ndarray)
  assert len(nhpp) == 10


def _test_mc_parallel(model, model_i):

  import mpi4py.MPI as mpi
  comm = mpi.COMM_WORLD

  # test model has identical streams
  mc = model.mc
  mc.reset()
  assert mc.seed() == 19937

  a = mc.ustream(5)
  all_a = comm.gather(a, root=0)
  all_states = comm.gather(mc.state(), root=0)

  if no.mpi.rank() == 0:
    for r in range(0, no.mpi.size()):
      assert np.all(all_states[0] == all_states[r])
      assert np.all(a - all_a[r] == 0.0)

  # test model_i has independent streams
  mc = model_i.mc
  mc.reset()
  assert mc.seed() == 19937 + no.mpi.rank()

  a = mc.ustream(5)
  all_a = comm.gather(a, root=0)
  all_states = comm.gather(mc.state(), root=0)

  # check all other streams different
  if no.mpi.rank() == 0:
    for r in range(1, no.mpi.size()):
      if platform.system() != "Darwin":
        # mc.state() just returns 0 on OSX due to an apparent bug in MT19937 that intermittently segfaults
        assert all_states[r] != all_states[0]
      assert not np.all(a - all_a[r] == 0.0)

