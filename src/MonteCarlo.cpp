
#include "MonteCarlo.h"
#include "ArrayHelpers.h"
#include "Log.h"
#include "Module.h"
#include "Timeline.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numpy/random/bitgen.h>
#include <random>
#include <sstream>

namespace {

void validate_prob(double prob) {
  if (std::isnan(prob) || prob < 0.0 || prob > 1.0) {
    throw no::value_error("probabilities must be in [0,1], got %%"s % prob);
  }
}

void validate_prob(const no::ArrayLike<double>& prob) {
  std::for_each(no::cbegin(prob), no::cend(prob), [](double p) { validate_prob(p); });
}

void validate_lambda(const no::ArrayLike<double>& lambda) {
  std::for_each(no::cbegin(lambda), no::cend(lambda), [](double l) {
    if (std::isnan(l) || l < 0.0) {
      throw no::value_error("intensities must be >=0, got %%"s % l);
    }
  });
}

// the arrival processes take the largest intensity, and extrapolate the final one
void validate_lambda_t(const no::ArrayLike<double>& lambda_t) {
  if (lambda_t.size() == 0) {
    throw no::value_error("intensities must not be empty");
  }
  validate_lambda(lambda_t);
}

} // namespace

// helper functions for basic seeding strategies

// independent streams per-processs (but threads within a process will have identical streams)
int32_t no::MonteCarlo::deterministic_independent_stream() noexcept {
  return 19937 + no::env::rank.load(std::memory_order_relaxed);
}

int32_t no::MonteCarlo::deterministic_identical_stream() noexcept { return 19937; }

int32_t no::MonteCarlo::nondeterministic_stream() {
  thread_local std::random_device rand;
  return rand();
}

no::MonteCarlo::MonteCarlo(std::function<int32_t()> seeder)
    : m_seeder(std::move(seeder)), m_seed(m_seeder()), m_prng(m_seed) {}

int32_t no::MonteCarlo::seed() const noexcept { return m_seed; }

void no::MonteCarlo::init_bitgen(nb::capsule capsule) {
  bitgen_t* bitgen = static_cast<bitgen_t*>(capsule.data());
  bitgen->state = this;
  bitgen->next_uint64 = [](void* p) { return static_cast<no::MonteCarlo*>(p)->raw(); };
  bitgen->next_uint32 = [](void* p) { return static_cast<uint32_t>(static_cast<no::MonteCarlo*>(p)->raw()); };
  bitgen->next_double = [](void* p) { return static_cast<no::MonteCarlo*>(p)->u01(); };
  bitgen->next_raw = [](void* p) { return static_cast<no::MonteCarlo*>(p)->raw(); };
}

void no::MonteCarlo::reset() {
  m_seed = m_seeder();
  m_prng.seed(m_seed);
}

std::string no::MonteCarlo::repr() const noexcept { return "<neworder.MonteCarlo seed=%%>"s % seed(); }

size_t no::MonteCarlo::state() const noexcept {
  std::ostringstream s;
  s << m_prng;
  return std::hash<std::string>{}(s.str());
}

// raw unsigned 64-bit ints (for enabling numpy to use this generator)
uint64_t no::MonteCarlo::raw() noexcept {
  return (static_cast<uint64_t>(m_prng()) << 32) | static_cast<uint64_t>(m_prng());
}

// uniform [0,1)
double no::MonteCarlo::u01() noexcept {
  static const double SCALE = 1.0 / (1ull << 32);
  return m_prng() * SCALE;
}

no::NpArray<double> no::MonteCarlo::ustream(Py_ssize_t n) {
  return no::make_array<double>({n}, [this]() { return u01(); });
}

no::NpArray<int64_t> no::MonteCarlo::sample(Py_ssize_t n, const ArrayLike<double>& cat_weights) {
  size_t m = cat_weights.size();
  const double* p = no::cbegin(cat_weights);

  std::vector<double> cumul = no::cumulative(p, m);

  return no::make_array<int64_t>({n}, [this, &cumul]() {
    double r = u01();
    int64_t k = 0;
    while (cumul[k] < r)
      ++k;
    return k;
  });
}

// simple hazard constant probability
no::NpArray<double> no::MonteCarlo::hazard(double prob, Py_ssize_t n) {
  validate_prob(prob);
  return no::make_array<double>({n}, [&]() { return (u01() < prob) ? 1.0 : 0.0; });
}

// hazard with varying probablities
no::NpArray<double> no::MonteCarlo::hazard(const ArrayLike<double>& prob) {
  validate_prob(prob);
  return no::unary_op<double, double>(prob, [this](double p) { return u01() < p ? 1.0 : 0.0; });
}

// computes stopping times
no::NpArray<double> no::MonteCarlo::stopping(double prob, Py_ssize_t n) {
  validate_prob(prob);
  double rprob = 1.0 / prob;

  // if p=0 will return no::time::far_future() (NOT no::time::never())
  // the argument being that stopping is an event that must happen, as opposed to an arrival, which might not
  return no::make_array<double>({n}, [&]() { return -::log(u01()) * rprob; });
}

no::NpArray<double> no::MonteCarlo::stopping(const ArrayLike<double>& prob) {
  validate_prob(prob);
  return no::unary_op<double, double>(prob, [this](double p) { return -::log(u01()) / p; });
}

// multiple-arrival (0+) process
no::NpArray<double> no::MonteCarlo::arrivals(const ArrayLike<double>& lambda_t, double dt, Py_ssize_t n, double gap) {
  validate_lambda_t(lambda_t);
  const double* pl = no::cbegin(lambda_t);
  size_t nl = lambda_t.size();

  // validate lambdas - but what exactly is valid?
  if (pl[nl - 1] != 0.0) {
    throw nb::value_error("Multiple-arrival Non-homogeneous Poisson process requires a zero final hazard rate");
  }

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  std::vector<std::vector<double>> times(n);

  double tmax = (nl - 1) * dt;
  size_t imax = 0;

  for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
    // rejection sampling
    double pt = 0.0;
    do {
      do {
        pt += -::log(u01()) / lambda_u;
        // final entry in lambda_t is flat extrapolated...
        lambda_i = pl[std::min((size_t)(pt / dt), nl - 1)];
        if (pt > tmax && lambda_i == 0.0) {
          pt = no::time::never();
          break;
        }
      } while (u01() > lambda_i / lambda_u);
      times[i].push_back(pt);
      pt += gap;
    } while (pt < tmax);
    imax = std::max(times[i].size(), imax);
    // no::log("%%: %%"s % i % times[i]);
  }

  // each row's final entry is dropped, so with n=0 (and so imax=0) avoid underflowing the column count
  const size_t ncols = imax > 0 ? imax - 1 : 0;
  NpArray<double> nptimes = no::allocate<double>(Index_t<2>{n, static_cast<Py_ssize_t>(ncols)});
  double* pa = nptimes.data();
  std::fill(pa, pa + nptimes.size(), no::time::never());

  for (size_t i = 0; i < times.size(); ++i) {
    for (size_t j = 0; j < times[i].size() - 1; ++j) {
      pa[j] = times[i][j];
    }
    pa += ncols;
  }

  return nptimes;
}

no::NpArray<double> no::MonteCarlo::first_arrival(const ArrayLike<double>& lambda_t, double dt, Py_ssize_t n,
                                                  double minval) {
  validate_lambda_t(lambda_t);
  const double* pl = no::cbegin(lambda_t);
  size_t nl = lambda_t.size();

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);

  double lambda_i;

  NpArray<double> times = no::allocate<double>(n);
  double* pt = times.data();
  double tmax = (nl - 1) * dt;

  for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
    // rejection sampling
    pt[i] = minval;
    do {
      pt[i] += -::log(u01()) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[std::min((size_t)(pt[i] / dt), nl - 1)];
      // deal with open case (event not certain to happen)
      if (pt[i] > tmax && lambda_i == 0.0) {
        pt[i] = no::time::never();
        break;
      }
    } while (u01() > lambda_i / lambda_u);
  }
  return times;
}

// next-arrival process - times of transition from a state arrived at at startingpoints to a subsequent state, with an
// optional deterministic minimum separation if the state hasn't been arrived at (no::never())
no::NpArray<double> no::MonteCarlo::next_arrival(const ArrayLike<double>& startingpoints,
                                                 const ArrayLike<double>& lambda_t, double dt, bool relative,
                                                 double minsep) {
  validate_lambda_t(lambda_t);
  size_t n = startingpoints.size();
  const double* ps = no::cbegin(startingpoints);

  const double* pl = no::cbegin(lambda_t);
  size_t nl = lambda_t.size();
  double tmax = (nl - 1) * dt;

  // What is the optimal lambda_u? For now largest value
  double lambda_u = *std::max_element(pl, pl + nl);
  double lambda_i;

  NpArray<double> times = no::allocate<double>(static_cast<Py_ssize_t>(n));
  double* pt = times.data();

  for (size_t i = 0; i < n; ++i) {
    // account for any deterministic time lag (e.g. 9 months between births)
    double offset = ps[i] + minsep;
    // skip if we haven't actually arrived at the state to transition from
    if (no::time::isnever(offset)) {
      pt[i] = no::time::never();
      continue;
    }
    // offset if lambdas in absolute time (not relative to start point)
    pt[i] = relative ? 0.0 : offset;
    do {
      pt[i] += -::log(u01()) / lambda_u;
      // final entry in lambda_t is flat extrapolated...
      lambda_i = pl[std::min((size_t)(pt[i] / dt), nl - 1)];
      if (pt[i] > tmax && lambda_i == 0.0) {
        pt[i] = no::time::never();
        break;
      }
    } while (u01() > lambda_i / lambda_u);
    // restore offset if required
    pt[i] += relative ? offset : 0.0;
  }
  return times;
}

no::NpArray<int64_t> no::MonteCarlo::counts(const ArrayLike<double>& lambda, double dt) {
  validate_lambda(lambda);

  size_t n = lambda.size();
  NpArray<int64_t> counts = no::allocate<int64_t>(static_cast<Py_ssize_t>(n));
  int64_t* pc = counts.data();

  for (const double* it = no::cbegin(lambda); it != no::cend(lambda); ++it, ++pc) {
    // sample arrivals within [0,dt)
    *pc = 0;
    double t = -std::log(u01()) / *it;
    while (t < dt) {
      ++*pc;
      t += -std::log(u01()) / *it;
    }
  }
  return counts;
}

//
std::vector<double> no::cumulative(const double* p, size_t n) {
  static const double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());

  if (n == 0)
    throw no::value_error("probabilities must not be empty");

  std::vector<double> cumprobs(n);
  if (p[0] < 0.0 || p[0] > 1.0)
    throw no::value_error("invalid probability: %% at index 0"s % p[0]);

  cumprobs[0] = p[0];
  for (size_t i = 1; i < n; ++i) {
    if (p[i] < 0.0 || p[i] > 1.0)
      throw no::value_error("invalid probability %% at index %%"s % p[i] % i);
    cumprobs[i] = cumprobs[i - 1] + p[i];
  }

  // check probabilities sum to unity within tolerance
  if (fabs(cumprobs[n - 1] - 1.0) > tolerance)
    throw no::value_error("cumulative probabilities don't sum to unity: %%"s % cumprobs);
  // then force the final value to 1 (in case its slightly out, which could cause a problem)
  cumprobs[n - 1] = 1.0;

  return cumprobs;
}

size_t no::interp(const std::vector<double>& cumprob, double x) {
  // TODO would bisection search here be better (even for small arrays)?
  size_t lbound = 0;
  for (; lbound < cumprob.size() - 1; ++lbound) {
    if (cumprob[lbound] > x)
      break;
  }
  return lbound;
}
