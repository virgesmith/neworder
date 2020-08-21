#include "NPArray.h"
#include "Timeline.h"

// the vectorised (nparray) implementation of Timeline::isnever
py::array no::isnever(const py::array& x)
{
  return no::unary_op<bool, double>(x, Timeline::isnever);
}

py::array no::logistic(const py::array& x, double x0, double k)
{
  return no::unary_op<double, double>(x, [&](double x) { return 1.0 / (1.0 + exp(-k*(x-x0))); });
}

py::array no::logit(const py::array& x)
{
  // NB no check for x not in [0,1)
  return no::unary_op<double, double>(x, [](double x) { return log(x/(1.0 - x)); });
}