#include "SplitMix64.h"
#include "ArrayHelpers.h"
#include "Log.h"

// SplitMix64 finalizer - Stafford Variant 13 mixing constants. Do not change.
namespace {

constexpr uint64_t SM_MULT1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t SM_MULT2 = 0x94D049BB133111EBULL;
constexpr double INV_2_53 = 1.0 / (1ULL << 53);

uint64_t splitmix64(uint64_t z) noexcept {
  z = (z ^ (z >> 30)) * SM_MULT1;
  z = (z ^ (z >> 27)) * SM_MULT2;
  return z ^ (z >> 31);
}

} // namespace

no::SplitMix64::SplitMix64(std::function<int64_t()> seeder, bool use_counter) noexcept
    : m_seeder(std::move(seeder)), m_use_counter(use_counter), m_counter(0) {}

uint64_t no::SplitMix64::counter() const noexcept { return m_counter.load(std::memory_order_relaxed); }

void no::SplitMix64::reset() noexcept { m_counter.store(0, std::memory_order_relaxed); }

std::string no::SplitMix64::repr() const {
  using namespace std::literals;
  if (m_use_counter)
    return "<neworder.SplitMix64 counter=%%>"s % m_counter.load(std::memory_order_relaxed);
  return "<neworder.SplitMix64>"s;
}

int64_t no::SplitMix64::hash64(std::string_view s) noexcept {
  // FNV-1a accumulates the string bytes then the SplitMix64 finalizer diffuses the bits.
  uint64_t h = 14695981039346656037ULL; // FNV-64 offset basis
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL; // FNV-64 prime
  }
  return static_cast<int64_t>(splitmix64(h));
}

no::SplitMix64::KeyMix no::SplitMix64::mix_keys(const py::args& args, const char* caller) {
  if (args.empty())
    throw py::value_error(std::string(caller) + " requires at least one argument");

  // Parse each arg into a flat value buffer. Array args also record their size
  // so we can build the output shape and compute multi-dimensional indices.
  struct Arg {
    std::vector<uint64_t> values; // size == 1 for scalars
    bool is_array;
  };

  std::vector<Arg> parsed;
  parsed.reserve(args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    py::handle obj = args[i];
    if (py::isinstance<py::int_>(obj)) {
      parsed.push_back({{static_cast<uint64_t>(obj.cast<int64_t>())}, false});
    } else {
      auto arr = py::array_t<int64_t, py::array::forcecast>::ensure(obj);
      if (!arr || arr.ndim() != 1)
        throw py::type_error(std::string(caller) + ": each argument must be a scalar int or a 1-D integer array");
      std::vector<uint64_t> vals(arr.size());
      const int64_t* ptr = arr.data();
      for (py::ssize_t j = 0; j < arr.size(); ++j)
        vals[j] = static_cast<uint64_t>(ptr[j]);
      parsed.push_back({std::move(vals), true});
    }
  }

  KeyMix mix;

  // Premix all scalar args (in argument order) into a single salt, computed
  // once per call. The counter (if enabled) is folded in last.
  mix.salt = static_cast<uint64_t>(m_seeder());
  for (const auto& arg : parsed)
    if (!arg.is_array)
      mix.salt = splitmix64(mix.salt ^ arg.values[0]);
  if (m_use_counter)
    mix.salt = splitmix64(mix.salt ^ m_counter.fetch_add(1, std::memory_order_relaxed));

  // Collect the array axes (in argument order) - they define the output shape
  // and the per-element hash steps applied on top of the salt.
  for (auto& arg : parsed) {
    if (arg.is_array) {
      mix.shape.push_back(static_cast<py::ssize_t>(arg.values.size()));
      mix.axes.push_back(std::move(arg.values));
    }
  }

  // Row-major strides over the output dimensions.
  const size_t ndim = mix.shape.size();
  mix.strides.assign(ndim, 1);
  for (int k = static_cast<int>(ndim) - 2; k >= 0; --k)
    mix.strides[k] = mix.strides[k + 1] * mix.shape[k + 1];

  mix.size = 1;
  for (auto s : mix.shape)
    mix.size *= s;

  return mix;
}

uint64_t no::SplitMix64::hash_at(const KeyMix& mix, py::ssize_t flat) noexcept {
  uint64_t h = mix.salt;
  for (size_t k = 0; k < mix.shape.size(); ++k) {
    const py::ssize_t ik = (flat / mix.strides[k]) % mix.shape[k];
    h = splitmix64(h ^ mix.axes[k][static_cast<size_t>(ik)]);
  }
  return h;
}

py::array_t<double> no::SplitMix64::uarray(py::args raw_args) {
  const KeyMix mix = mix_keys(raw_args, "uarray");

  py::array_t<double> result(mix.shape); // shape={} produces a 0-d array for all-scalar args
  double* out = result.mutable_data();

  for (py::ssize_t flat = 0; flat < mix.size; ++flat)
    out[flat] = static_cast<double>(hash_at(mix, flat) >> 11) * INV_2_53;

  return result;
}

py::array_t<int64_t> no::SplitMix64::raw(py::args raw_args) {
  const KeyMix mix = mix_keys(raw_args, "raw");

  py::array_t<int64_t> result(mix.shape); // shape={} produces a 0-d array for all-scalar args
  int64_t* out = result.mutable_data();

  // The full 64 bits, reinterpreted (not clamped) as signed - uarray() discards the
  // low 11 in exchange for a uniform mapping onto float64's 53-bit mantissa.
  for (py::ssize_t flat = 0; flat < mix.size; ++flat)
    out[flat] = static_cast<int64_t>(hash_at(mix, flat));

  return result;
}
