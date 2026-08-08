#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace no {

class NEWORDER_EXPORT SplitMix64 final {
public:
  explicit SplitMix64(std::function<int64_t()> seeder, bool use_counter = false) noexcept;

  uint64_t counter() const noexcept;
  void reset() noexcept; // resets the counter only; seeder is called fresh on each uarray()/raw()

  py::array_t<double> uarray(py::args args);

  // The unmapped 64-bit hashes underlying uarray(), for seeding other generators or
  // deriving variates uarray() can't express. Identical keying, shape and counter
  // semantics to uarray(); only the mapping of the hash to the output type differs.
  py::array_t<int64_t> raw(py::args args);

  static int64_t hash64(std::string_view s) noexcept;

  std::string repr() const;

private:
  // The per-call salt plus the output geometry, shared by uarray() and raw() so the
  // two cannot drift apart. One entry in axes/shape/strides per 1-D array argument.
  struct KeyMix {
    uint64_t salt;
    std::vector<std::vector<uint64_t>> axes;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    py::ssize_t size;
  };

  // Invokes the seeder and consumes a counter increment, so call exactly once per call.
  KeyMix mix_keys(const py::args& args, const char* caller);
  static uint64_t hash_at(const KeyMix& mix, py::ssize_t flat) noexcept;

  std::function<int64_t()> m_seeder;
  bool m_use_counter;
  std::atomic<uint64_t> m_counter;
};

} // namespace no
