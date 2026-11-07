#pragma once

#include "ArrayHelpers.h"
#include "NewOrder.h"

// "vectorised" functions operating on, and/or returning numpy arrays, using the generic helper functions in
// ArrayHelpers.h

namespace no {

namespace time {
NEWORDER_EXPORT NpArray<bool> isnever_a(const ArrayLike<double>& x);
}
// logistic function f(x; x0, k)
NEWORDER_EXPORT NpArray<double> logistic(const ArrayLike<double>& x, double x0, double k);

// logit function
NEWORDER_EXPORT NpArray<double> logit(const ArrayLike<double>& x);

} // namespace no
