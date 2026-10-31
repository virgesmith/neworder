#include "Timeline.h"

#include <cmath>
#include <limits>

// returns a floating point number that compares less than any other number
double no::time::distant_past() { return -std::numeric_limits<double>::infinity(); }

// returns a floating point number that compares greater than any other number
double no::time::far_future() { return std::numeric_limits<double>::infinity(); }

// returns a floating point number that compares unequal to (and unordered w.r.t) any other number
// thus the following all evaluate to true: never() != never(), !(x < never()), !(x >= never()) (so be careful!)
double no::time::never() { return std::numeric_limits<double>::quiet_NaN(); }

// use this rather than direct comparison to never, as NaN != NaN (as above)
bool no::time::isnever(double t) { return std::isnan(t); }
