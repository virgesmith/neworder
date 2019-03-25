
#include "Module.h"

#include "Version.h"
#include "Timeline.h"
#include "Environment.h"
#include "Inspect.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "MPIComms.h"


#include "NewOrder.h"

#include <pybind11/embed.h>

#include <iostream>

namespace {

// not visible to (rest of) C++ - use function declareds in Log.h
void log_obj(const py::object& msg)
{
  std::cout << neworder::getenv().context(neworder::Environment::PY) << pycpp::as_string(msg.ptr()) << std::endl;
}

}

neworder::Callback neworder::Callback::eval(const std::string& code)
{
  return Callback(code, false);
}

neworder::Callback neworder::Callback::exec(const std::string& code)
{
  return Callback(code, true);
}

neworder::Callback::Callback(const std::string& code, bool exec/*, const std::string& locals*/) : m_exec(exec), m_code(code)
{
  // assuming they ref current env
  m_globals = py::module::import("__main__").attr("__dict__");
  m_locals = py::module::import("neworder").attr("__dict__");
}

py::object neworder::Callback::operator()() const 
{
  py::scoped_interpreter guard{};
  // evaluate the global/local namespaces at the last minute? or do they update dynamically?
  if (m_exec)
  {
    py::exec(m_code.c_str(), m_globals, m_locals);
    return py::object();
  }
  else
  {
    return py::eval(m_code.c_str(), m_globals, m_locals);
  }
}

const char* neworder::module_name()
{
  return "neworder";
}

const char* neworder::module_version()
{
  return NEWORDER_VERSION_STRING;
}

std::string neworder::python_version()
{
  return neworder::getenv().version();
}

void neworder::shell(/*const py::object& local*/)
{
  if (neworder::getenv().size() != 1) 
  {
    neworder::log("WARNING: shell disabled in parallel mode, ignoring");
    return;
  }
  py::dict kwargs;
  kwargs["banner"] = py::str("[starting neworder debug shell]");
  kwargs["exitmsg"] = py::str("[exiting neworder debug shell]");
  //py::module::import("neworder");
  //kwargs["local"] = py::handle<>(PyObject_Dir());
  py::object interpreter = py::module::import("code").attr("interact")(*py::tuple(), **kwargs);
}


// TODO move? (this is called from run.cpp but not exposed to python) 
void neworder::set_timeline(const py::tuple& spec) 
{
  size_t n = py::len(spec);
  std::vector<double> checkpoint_times(n - 1);
  for (size_t i = 0; i < n - 1; ++i)
  {
    // allow integer (or float) values
    py::int_ intval();
    if (py::isinstance<py::int_>(spec[i]))
    {
      checkpoint_times[i] = spec[i].cast<int>();
    }
    else
    {
      checkpoint_times[i] = spec[i].cast<double>();
    }
  }

  size_t nsteps = spec[n-1].cast<int>();

  getenv().timeline() = Timeline(checkpoint_times, nsteps);
}

// Deal with defaulted arguments for certain functions
// see https://stackoverflow.com/questions/35886682/passing-specific-arguments-to-boost-python-function-with-default-arguments

// np::array first_arrival(const np::array&, double, size_t, double = 0.0);
//BOOST_PYTHON_FUNCTION_OVERLOADS(first_arrival_default, neworder::first_arrival, 3, 4)
// np::array next_arrival(const np::array&, const np::array&, double, bool = false, double = 0.0);
//BOOST_PYTHON_FUNCTION_OVERLOADS(next_arrival_default, neworder::next_arrival, 3, 5)


// python-visible log function defined above

PYBIND11_EMBEDDED_MODULE(neworder_, m)
{
  namespace no = neworder;

  // utility/diagnostics
  m.def("name", no::module_name);
  m.def("version", no::module_version);
  m.def("python", no::python_version);
  m.def("log", log_obj);
  m.def("shell", no::shell);
  m.def("reseed", neworder::Environment::reset);

  // time-related
  //m.def("set_timeline", no::set_timeline);
  m.def("distant_past", no::Timeline::distant_past);
  m.def("far_future", no::Timeline::far_future);
  m.def("never", no::Timeline::never);
  m.def("isnever", no::Timeline::isnever); // scalar 
  m.def("isnever", no::nparray::isnever); // array
  
  // MC
  m.def("ustream", no::ustream);
  // explicitly give function type for overloads 
  m.def<np::array (*)(double, size_t)>("hazard", no::hazard);
  m.def<np::array (*)(const np::array&)>("hazard", no::hazard);
  m.def<np::array (*)(double, size_t)>("stopping", no::stopping);
  m.def<np::array (*)(const np::array&)>("stopping", no::stopping);
  //m.def("stopping_nhpp", no::stopping_nhpp);
  m.def("arrivals", no::arrivals);
  // deal with default minval arg - see above
  m.def("first_arrival", no::first_arrival); //, first_arrival_default(py::args("lambda_t", "dt", "n", "minval")));
  m.def("next_arrival", no::next_arrival); //, next_arrival_default(py::args("startingpoints", "lambda_t", "delta_t", "relative", "minsep")));

  m.def("lazy_exec", no::Callback::exec);
  m.def("lazy_eval", no::Callback::eval);

  // working on pandas df manipulation  
  m.def("transition", no::df::transition);
  m.def("directmod", no::df::directmod);
  m.def("linked_change", no::df::linked_change /*, py::return_value_policy<py::return_by_value>()*/);

  // MPI
  m.def("rank", neworder::Environment::rank);
  m.def("size", neworder::Environment::size);
  m.def("send", no::mpi::send_obj);
  m.def("receive", no::mpi::receive_obj);
  m.def("send_csv", no::mpi::send_csv);
  m.def("receive_csv", no::mpi::receive_csv);
  m.def("broadcast", no::mpi::broadcast_obj);
  m.def("gather", no::mpi::gather_array);
  m.def("scatter", no::mpi::scatter_array);
  m.def("allgather", no::mpi::allgather_array/*, py::return_value_policy<py::return_by_value>()*/);
  m.def("sync", no::mpi::sync);
  m.def("indep", neworder::Environment::indep);
  
  // Deferred eval/exec of Python code
  py::class_<no::Callback>(m, "Callback"/*, py::no_init*/)
    .def("__call__", &no::Callback::operator())
    .def("__str__", &no::Callback::code)
    ;

  // Example of wrapping an STL container
  // py::class_<std::vector<double>>("DVector", py::init<int>())
  //   .def("__len__", &std::vector<double>::size)
  //   .def("clear", &std::vector<double>::clear)
  //   .def("__getitem__", &vector_get<double>/*, py::return_value_policy<py::copy_non_const_reference>()*/)
  //   .def("__setitem__", &vector_set<double>, py::with_custodian_and_ward<1,2>()) // to let container keep value
  //   .def("__str__", &no::vector_to_string<double>)
  //   .def("tolist", &no::vector_to_py_list<double>)
  //   .def("fromlist", &no::py_list_to_vector<double>)
  //   // operators
  //   .def(py::self + double())
  //   .def(double() + py::self)
  //   .def(py::self * double())
  //   .def(double() * py::self)
  //   ;  

}

void neworder::import_module()
{
  // TODO anything required?
  // First register callback module
  //PyImport_AppendInittab(module_name(), &PyInit_neworder);
}

