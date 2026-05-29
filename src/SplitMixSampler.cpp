#include "SplitMixSampler.h"
#include "ArrayHelpers.h"

// SplitMix64 finalizer — Stafford Variant 13 mixing constants. Do not change.
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

no::SplitMixSampler::SplitMixSampler(std::function<int64_t()> seeder, bool use_counter) noexcept
    : m_seeder(std::move(seeder)), m_use_counter(use_counter), m_counter(0) {}

int64_t no::SplitMixSampler::seed() const { return m_seeder(); }

uint64_t no::SplitMixSampler::counter() const noexcept { return m_counter; }

void no::SplitMixSampler::reset() noexcept { m_counter = 0; }

std::string no::SplitMixSampler::repr() const { return "SplitMixSampler(seed=" + std::to_string(m_seeder()) + ")"; }

py::array_t<double> no::SplitMixSampler::uarray(py::args raw_args) const {
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
  // once per call. This matches the Python original's structure: scalars form a
  // "context hash" (module, year, draw index, …) that is independent of which
  // array elements are present, and the counter (if enabled) is folded in first.
  uint64_t salt = static_cast<uint64_t>(m_seeder());
  for (const auto& ax : axes)
    if (!ax.is_array)
      salt = splitmix64(salt ^ ax.values[0]);
  if (m_use_counter)
    salt = splitmix64(salt ^ m_counter++);

  // Collect the array axes (in argument order) — they define the output shape
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
