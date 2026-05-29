#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>

#include <cstdint>
#include <functional>

namespace no {

class NEWORDER_EXPORT SplitMixSampler {
public:
  explicit SplitMixSampler(std::function<int64_t()> seeder, bool use_counter = false) noexcept;

  int64_t seed() const; // calls the seeder; not noexcept as the callable may throw
  uint64_t counter() const noexcept;
  void reset() noexcept; // resets the counter; seeder is called fresh on each uarray()

  py::array_t<double> uarray(py::args args) const;

  std::string repr() const; // calls the seeder; not noexcept as the callable may throw

private:
  std::function<int64_t()> m_seeder;
  bool m_use_counter;
  mutable uint64_t m_counter;
};

} // namespace no
