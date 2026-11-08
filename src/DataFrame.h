#pragma once

#include "ArrayHelpers.h"
#include "NewOrder.h"
// Data frame manipulation routines

namespace no {

class MonteCarlo;

namespace df {

NpArray<int64_t> unique_index(size_t n);

nb::object transition(no::MonteCarlo& mc, const ArrayLike<double>& matrix, const nb::object& series);

nb::object transition_conditional(no::MonteCarlo& mc, const nb::dict& matrices, const nb::object& group,
                                  const nb::object& series);

} // namespace df

} // namespace no
