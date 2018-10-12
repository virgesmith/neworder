
#include "Module.h"

#include "Version.h"
#include "Timeline.h"
#include "Environment.h"
#include "Inspect.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "MPIComms.h"


#include "python.h"

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
  m_globals = py::import("__main__").attr("__dict__");
  m_locals = py::import("neworder").attr("__dict__");
}

py::object neworder::Callback::operator()() const 
{
  // see https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/reference/embedding.html#embedding.boost_python_exec_hpp
  // evaluate the global/local namespaces at the last minute? or do they update dynamically?
  if (m_exec)
  {
    return py::exec(m_code.c_str(), m_globals, m_locals);
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
  //py::import("neworder");
  //kwargs["local"] = py::handle<>(PyObject_Dir());
  py::object interpreter = py::import("code").attr("interact")(*py::tuple(), **kwargs);
}


// TODO move? (this is called from run.cpp but not exposed to python) 
void neworder::set_timeline(const py::tuple& spec) 
{
  size_t n = py::len(spec);
  std::vector<double> checkpoint_times(n - 1);
  for (size_t i = 0; i < n - 1; ++i)
  {
    // allow integer (or float) values
    py::extract<int> intval(spec[i]);
    if (intval.check())
    {
      checkpoint_times[i] = intval();
    }
    else
    {
      checkpoint_times[i] = py::extract<double>(spec[i]);
    }
  }

  size_t nsteps = py::extract<int>(spec[n-1]);

  getenv().timeline() = Timeline(checkpoint_times, nsteps);
}


// python-visible log function defined above

BOOST_PYTHON_MODULE(neworder)
{
  namespace no = neworder;

  // utility/diagnostics
  py::def("name", no::module_name);
  py::def("version", no::module_version);
  py::def("python", no::python_version);
  py::def("log", log_obj);
  py::def("shell", no::shell);
  py::def("reseed", neworder::Environment::reset);

  // time-related
  //py::def("set_timeline", no::set_timeline);
  py::def("distant_past", no::Timeline::distant_past);
  py::def("far_future", no::Timeline::far_future);
  py::def("never", no::Timeline::never);
  py::def("isnever", no::Timeline::isnever); // scalar 
  py::def("isnever", no::nparray::isnever); // array
  
  // MC
  py::def("ustream", no::ustream);
  py::def("hazard", no::hazard);
  py::def("hazard_v", no::hazard_v);
  py::def("stopping", no::stopping);
  py::def("stopping_v", no::stopping_v);
  py::def("stopping_nhpp", no::stopping_nhpp);
  py::def("arrivals", no::arrivals);
  py::def("first_arrival", no::first_arrival);
  py::def("next_arrival", no::next_arrival);

  py::def("lazy_exec", no::Callback::exec);
  py::def("lazy_eval", no::Callback::eval);

  // working on pandas df manipulation  
  py::def("transition", no::df::transition);
  py::def("directmod", no::df::directmod);
  py::def("append", no::df::append, py::return_value_policy<py::return_by_value>());

  // MPI
  py::def("rank", neworder::Environment::rank);
  py::def("size", neworder::Environment::size);
  py::def("send", no::mpi::send_obj);
  py::def("receive", no::mpi::receive_obj);
  py::def("send_csv", no::mpi::send_csv);
  py::def("receive_csv", no::mpi::receive_csv);
  py::def("broadcast", no::mpi::broadcast_obj);
  py::def("gather", no::mpi::gather_array);
  py::def("scatter", no::mpi::scatter_array);
  py::def("allgather", no::mpi::allgather_array/*, py::return_value_policy<py::return_by_value>()*/);
  py::def("sync", no::mpi::sync);
  py::def("indep", neworder::Environment::indep);
  
  // Deferred eval/exec of Python code
  py::class_<no::Callback>("Callback", py::no_init)
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
  // First register callback module
  PyImport_AppendInittab(module_name(), &PyInit_neworder);
}

