#include "NPArray.h"

#include "Timeline.h"

// the vectorised (nparray) implementation of Timerline::isnever
np::array no::nparray::isnever(const np::array& x)
{
  size_t n = x.size();
  np::array result = np::empty_1d_array<bool>(n);

  bool* const pr = np::begin<bool>(result);
  const double* const px = np::cbegin<double>(x);
  for (size_t i = 0; i < n; ++i)
  {
    pr[i] = no::Timeline::isnever(px[i]);
  }
  return result;
}
