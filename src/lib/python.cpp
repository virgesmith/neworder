
#include "python.h"

std::ostream& operator<<(std::ostream& os, const py::object& o)
{
  return os << py::extract<std::string>(py::str(o))();
}

// std::ostream& operator<<(std::ostream& os, const np::ndarray& a)
// {
//   return os << py::extract<std::string>(py::str(a))();
// }
