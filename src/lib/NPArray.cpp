#include "NPArray.h"

#include "Timeline.h"

// the vectorised (nparray) implementation of Timeline::isnever
py::array no::isnever(const py::array& x)
{
  py::buffer_info bufin = x.request();
  std::vector<ssize_t> shape(x.shape(), x.shape() + x.ndim());
  py::array_t<bool> result(shape);
  py::buffer_info bufout = result.request();

  double *p = (double*)bufin.ptr;
  bool* r = (bool*)bufout.ptr;
  for (size_t i = 0; i < bufin.size; ++i)
  {
    r[i] = no::Timeline::isnever(p[i]);
  }

  // size_t n = x.size();
  // py::array result = no::empty_1d_array<bool>(n);

  // bool* const pr = no::begin<bool>(result);
  // const double* const px = no::cbegin<double>(x);
  // for (size_t i = 0; i < n; ++i)
  // {
  //   pr[i] = no::Timeline::isnever(px[i]);
  // }
  return result;
}
