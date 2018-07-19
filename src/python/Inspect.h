#pragma once

namespace pycpp {

// perhaps pycpp::String?
std::vector<std::string> dir(PyObject* obj)
{
  auto attrs = pycpp::List(PyObject_Dir(obj)).toVector<std::string>();
  return attrs;
}

}