
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

#include <iostream>

namespace {

// not visible to (rest of) C++ - use function declareds in Log.h
void log_obj(const py::object& msg)
{
  std::cout << no::getenv().context(no::Environment::PY) << pycpp::as_string(msg.ptr()) << std::endl;
}

}

no::Callback no::Callback::eval(const std::string& code)
{
  return Callback(code, false);
}

no::Callback no::Callback::exec(const std::string& code)
{
  return Callback(code, true);
}

no::Callback::Callback(const std::string& code, bool exec/*, const std::string& locals*/) : m_exec(exec), m_code(code)
{
  // assuming they ref current env
  m_globals = py::module::import("__main__").attr("__dict__");
  m_locals = py::module::import("neworder").attr("__dict__");
}

py::object no::Callback::operator()() const 
{
  // evaluate the global/local namespaces at the last minute? or do they update dynamically?
  if (m_exec)
  {
    py::exec(m_code.c_str(), m_globals, m_locals);
    return py::none();
  }
  else
  {
    return py::eval(m_code.c_str(), m_globals, m_locals);
  }
}

const char* no::module_name()
{
  return "neworder";
}

const char* no::module_version()
{
  return NEWORDER_VERSION_STRING;
}

std::string no::python_version()
{
  return no::getenv().python_version();
}

void no::shell(/*const py::object& local*/)
{
  if (no::getenv().size() != 1) 
  {
    no::log("WARNING: shell disabled in parallel mode, ignoring");
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
void no::set_timeline(const py::tuple& spec) 
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


// python-visible log function defined above
PYBIND11_EMBEDDED_MODULE(neworder, m)
{

  // utility/diagnostics
  m.def("name", no::module_name);
  m.def("version", no::module_version);
  m.def("python", no::python_version);
  m.def("log", log_obj);
  m.def("shell", no::shell);
  m.def("reseed", no::Environment::reset);

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

  m.def("arrivals", no::arrivals);
  // deal with default minval arg
  m.def<np::array (*)(const np::array&, double, size_t)>("first_arrival", [](const np::array& lambda_t, double dt, size_t n) { 
                                                                            return no::first_arrival(lambda_t, dt, n, 0.0); });
  m.def<np::array (*)(const np::array&, double, size_t, double)>("first_arrival", no::first_arrival);
   
  // deal with default relative and minval args
  m.def<np::array (*)(const np::array&, const np::array&, double)>("next_arrival", [](const np::array& startingpoints, const np::array& lambda_t, double dt) { 
                                                                                    return no::next_arrival(startingpoints, lambda_t, dt, false, 0.0); }); 
  m.def<np::array (*)(const np::array&, const np::array&, double, bool)>("next_arrival", [](const np::array& startingpoints, const np::array& lambda_t, double dt, bool relative) { 
                                                                                    return no::next_arrival(startingpoints, lambda_t, dt, relative, 0.0); }); 
  m.def<np::array (*)(const np::array&, const np::array&, double, bool, double)>("next_arrival", no::next_arrival); 

  m.def("lazy_exec", no::Callback::exec);
  m.def("lazy_eval", no::Callback::eval);

  // working on pandas df manipulation  
  m.def("transition", no::df::transition);
  m.def("directmod", no::df::directmod);
  m.def("linked_change", no::df::linked_change, py::return_value_policy::take_ownership);

  // MPI
  m.def("rank", no::Environment::rank);
  m.def("size", no::Environment::size);
  m.def("send", no::mpi::send_obj);
  m.def("receive", no::mpi::receive_obj);
  m.def("send_csv", no::mpi::send_csv);
  m.def("receive_csv", no::mpi::receive_csv);
  m.def("broadcast", no::mpi::broadcast_obj);
  m.def("gather", no::mpi::gather_array);
  m.def("scatter", no::mpi::scatter_array);
  m.def("allgather", no::mpi::allgather_array, py::return_value_policy::take_ownership);
  m.def("sync", no::mpi::sync);
  m.def("indep", no::Environment::indep);
  
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

void no::import_module()
{
  // TODO anything required?
  // First register callback module
  //PyImport_AppendInittab(module_name(), &PyInit_neworder);
}

