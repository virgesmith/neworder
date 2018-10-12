#include "NPArray.h"

#include "Timeline.h"

// the vectorised (nparray) implementation of Timerline::isnever
np::ndarray neworder::nparray::isnever(const np::ndarray& x)
{
  size_t n = pycpp::size(x);
  np::ndarray result = pycpp::empty_1d_array<bool>(n);

  bool* const pr = pycpp::begin<bool>(result);
  const double* const px = pycpp::begin<double>(x);
  for (size_t i = 0; i < n; ++i)
  {
    pr[i] = neworder::Timeline::isnever(px[i]);
  }
  return result;
}