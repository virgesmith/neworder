
#include "Log.h"
#include "Helpers.h"
#include "Module.h"

std::string std::to_string(char c) { return std::string(1, c); }

std::string std::to_string(const std::string& v) { return v; }

std::string std::to_string(const nb::object& o) { return nb::str(o).c_str(); }

void no::log(const std::string& msg) {
  if (no::env::verbose)
    nb::print((no::env::logPrefix[no::env::Context::CPP] % env::thread_id() % msg).c_str());
}

void no::warn(const std::string& msg) {
  if (PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1) == -1)
    throw nb::python_error();
}