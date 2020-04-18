#include "NewOrder.h"
#include "NPArray.h"
#include "Timeline.h"

// the vectorised (nparray) implementation of Timeline::isnever
NEWORDER_EXPORT py::array no::isnever(const py::array& x)
{
  return no::unary_op<bool, double>(x, Timeline::isnever);
}

NEWORDER_EXPORT py::array no::logistic(const py::array& x)//, double x0, double k)
{
  double k = 1.0; double x0 = 0.0;
  return no::unary_op<double, double>(x, [&](double x) { return 1.0 / (1.0 + exp(-k*(x-x0))); });
}

NEWORDER_EXPORT py::array no::logit(const py::array& x)
{
  // NB no check for x not in [0,1)
  return no::unary_op<double, double>(x, [](double x) { return log(x/(1.0 - x)); });
}