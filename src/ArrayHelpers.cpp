#include "ArrayHelpers.h"
#include "Time.h"

#include <cmath>

namespace {

// intentionally never released: these outlive the module, and numpy isn't unloaded before process exit
PyObject* np_ndarray = nullptr;
PyObject* np_empty = nullptr;
PyObject* np_asarray = nullptr;
PyObject* np_array = nullptr;
PyObject* np_dtypes[3] = {nullptr, nullptr, nullptr};

PyObject* new_ref(nb::object obj) { return obj.release().ptr(); }

nb::handle dtype_handle(no::numpy::DType dtype) { return np_dtypes[static_cast<size_t>(dtype)]; }

} // namespace

void no::numpy::init() {
  nb::module_ np = nb::module_::import_("numpy");
  np_ndarray = new_ref(np.attr("ndarray"));
  np_empty = new_ref(np.attr("empty"));
  np_asarray = new_ref(np.attr("asarray"));
  np_array = new_ref(np.attr("array"));
  np_dtypes[static_cast<size_t>(DType::FLOAT64)] = new_ref(np.attr("dtype")("float64"));
  np_dtypes[static_cast<size_t>(DType::INT64)] = new_ref(np.attr("dtype")("int64"));
  np_dtypes[static_cast<size_t>(DType::BOOL)] = new_ref(np.attr("dtype")("bool"));
}

bool no::numpy::is_ndarray(nb::handle obj) noexcept {
  return PyObject_TypeCheck(obj.ptr(), reinterpret_cast<PyTypeObject*>(np_ndarray));
}

nb::object no::numpy::empty(std::span<const Py_ssize_t> shape, DType dtype) {
  nb::object shape_tuple = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(shape.size())));
  if (!shape_tuple.is_valid())
    throw nb::python_error();
  for (size_t i = 0; i < shape.size(); ++i) {
    PyObject* extent = PyLong_FromSsize_t(shape[i]);
    if (!extent)
      throw nb::python_error();
    if (PyTuple_SetItem(shape_tuple.ptr(), static_cast<Py_ssize_t>(i), extent) != 0)
      throw nb::python_error();
  }
  return nb::handle(np_empty)(shape_tuple, dtype_handle(dtype));
}

nb::object no::numpy::asarray(nb::handle obj, DType dtype) {
  using namespace nb::literals;
  return nb::handle(np_asarray)(obj, "dtype"_a = dtype_handle(dtype), "order"_a = "C");
}

nb::object no::numpy::copy(nb::handle obj, DType dtype) {
  using namespace nb::literals;
  return nb::handle(np_array)(obj, "dtype"_a = dtype_handle(dtype), "order"_a = "C", "copy"_a = true);
}

// the vectorised (nparray) implementation of Timeline::isnever
no::NpArray<bool> no::time::isnever_a(const ArrayLike<double>& x) {
  return no::unary_op<bool, double>(x, time::isnever);
}

