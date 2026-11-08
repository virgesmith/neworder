
#include "Log.h"
#include "Helpers.h"
#include "Module.h"

std::string std::to_string(char c) { return std::string(1, c); }

std::string std::to_string(const std::string& v) { return v; }

std::string std::to_string(const nb::object& o) {
  nb::str s(o);
  Py_ssize_t size = 0;
  if (const char* utf8 = PyUnicode_AsUTF8AndSize(s.ptr(), &size))
    return std::string(utf8, size);
  // not encodable as UTF-8 (e.g. a lone surrogate), but this is only ever used to build messages, so escape it
  PyErr_Clear();
  nb::bytes escaped = nb::borrow<nb::bytes>(s.attr("encode")("utf-8", "backslashreplace"));
  return std::string(escaped.c_str(), escaped.size());
}

void no::log(const std::string& msg) {
  if (no::env::verbose)
    nb::print((no::env::logPrefix[no::env::Context::CPP] % env::thread_id() % msg).c_str());
}

void no::warn(const std::string& msg) {
  if (PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1) == -1)
    throw nb::python_error();
}