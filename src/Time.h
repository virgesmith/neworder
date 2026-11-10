#pragma once

namespace no {

namespace time {

// returns a floating point number that compares unequal to (and unordered w.r.t) any other number
// thus the following all evaluate to true: never() != never(), !(x < never()), !(x >= never()) (so be careful!)
double never();

// this MUST be used to correctly compare against never since NaN != NaN
bool isnever(double t);

// returns a floating point number that compares less than any other number
double distant_past();

// returns a floating point number that compares greater than any other number
double far_future();
} // namespace time

} // namespace no
