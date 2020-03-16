#include "NPArray.h"

#include "Timeline.h"

// the vectorised (nparray) implementation of Timeline::isnever
py::array no::isnever(const py::array& x)
{
  return no::unary_op<bool, double>(x, Timeline::isnever);
}
