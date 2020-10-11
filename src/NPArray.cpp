#include "NPArray.h"
#include "ArrayHelpers.h"
#include "Timeline.h"

// the vectorised (nparray) implementation of Timeline::isnever
py::array_t<bool> no::time::isnever_a(const py::array_t<double>& x)
{
  return no::unary_op<bool, double>(x, time::isnever);
}

py::array_t<double> no::logistic(const py::array_t<double>& x, double x0, double k)
{
  return no::unary_op<double, double>(x, [&](double x) { return 1.0 / (1.0 + exp(-k*(x-x0))); });
}

py::array_t<double> no::logit(const py::array_t<double>& x)
{
  // NB no check for x not in [0,1)
  return no::unary_op<double, double>(x, [](double x) { return log(x/(1.0 - x)); });
}