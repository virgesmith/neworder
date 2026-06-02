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

py::array_t<double> no::SplitMix64::uarray(py::args raw_args) {
  if (raw_args.empty())
    throw py::value_error("uarray requires at least one argument");

  // Parse each arg into a flat value buffer. Array args also record their size
  // so we can build the output shape and compute multi-dimensional indices.
  struct Axis {
    std::vector<uint64_t> values; // size == 1 for scalars
    bool is_array;
  };

  std::vector<Axis> axes;
  axes.reserve(raw_args.size());

  for (size_t i = 0; i < raw_args.size(); ++i) {
    py::handle obj = raw_args[i];
    if (py::isinstance<py::int_>(obj)) {
      axes.push_back({{static_cast<uint64_t>(obj.cast<int64_t>())}, false});
    } else {
      auto arr = py::array_t<int64_t, py::array::forcecast>::ensure(obj);
      if (!arr || arr.ndim() != 1)
        throw py::type_error("uarray: each argument must be a scalar int or a 1-D integer array");
      std::vector<uint64_t> vals(arr.size());
      const int64_t* ptr = arr.data();
      for (py::ssize_t j = 0; j < arr.size(); ++j)
        vals[j] = static_cast<uint64_t>(ptr[j]);
      axes.push_back({std::move(vals), true});
    }
  }

  // Premix all scalar args (in argument order) into a single salt, computed
  // once per call. The counter (if enabled) is folded in last.
  uint64_t salt = static_cast<uint64_t>(m_seeder());
  for (const auto& ax : axes)
    if (!ax.is_array)
      salt = splitmix64(salt ^ ax.values[0]);
  if (m_use_counter)
    salt = splitmix64(salt ^ m_counter.fetch_add(1, std::memory_order_relaxed));

  // Collect the array axes (in argument order) - they define the output shape
  // and the per-element hash steps applied on top of the salt.
  std::vector<const std::vector<uint64_t>*> array_axes;
  std::vector<py::ssize_t> shape;
  for (const auto& ax : axes) {
    if (ax.is_array) {
      array_axes.push_back(&ax.values);
      shape.push_back(static_cast<py::ssize_t>(ax.values.size()));
    }
  }

  // Row-major strides over the output dimensions.
  const size_t ndim = shape.size();
  std::vector<py::ssize_t> strides(ndim, 1);
  for (int k = static_cast<int>(ndim) - 2; k >= 0; --k)
    strides[k] = strides[k + 1] * shape[k + 1];

  py::ssize_t total = 1;
  for (auto s : shape)
    total *= s;

  py::array_t<double> result(shape); // shape={} produces a 0-d array for all-scalar args
  double* out = result.mutable_data();

  for (py::ssize_t flat = 0; flat < total; ++flat) {
    uint64_t h = salt;
    for (size_t k = 0; k < ndim; ++k) {
      const py::ssize_t ik = (flat / strides[k]) % shape[k];
      h = splitmix64(h ^ (*array_axes[k])[static_cast<size_t>(ik)]);
    }
    out[flat] = static_cast<double>(h >> 11) * INV_2_53;
  }

  return result;
}
