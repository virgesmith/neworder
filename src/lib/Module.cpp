
#include "Module.h"

#include "Timeline.h"
#include "Model.h"
#include "Environment.h"
#include "Inspect.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "Log.h"

#include "NewOrder.h"

#include <iostream>

using namespace py::literals;

namespace {

// not visible to (rest of) C++ - use the function declared in Log.h
void log_obj(const py::object& msg)
{
  std::cout << no::getenv().context(no::Environment::Context::PY) << msg << std::endl;
}

}


no::Runtime::Runtime(const std::string& local_module) : m_local(local_module)
{
}

py::object no::Runtime::operator()(const std::tuple<std::string, CommandType>& cmd) const 
{
  // evaluate the local namespace at the last minute as they don't update dynamically?
  py::dict locals = py::module::import("__main__").attr("__dict__");
  if (!m_local.empty())
  {
    //locals = py::module::import(m_local.c_str()).attr("__dict__");
    locals[m_local.c_str()] = py::module::import(m_local.c_str());
  }

  if (std::get<1>(cmd) == CommandType::Exec)
  {
    py::exec(std::get<0>(cmd).c_str(), py::globals(), locals);
    return py::none();
  }
  else
  {
    return py::eval(std::get<0>(cmd).c_str(), py::globals(), locals);
  }
}

// jump through hoops as msvc seems to strip quotes from defines, so need to add them here
#define STR2(x) #x
#define STR(x) STR2(x)
const char* no::module_version()
{
  return STR(NEWORDER_VERSION);
}

std::string no::python_version()
{
  return no::getenv().python_version();
}

// python-visible log function defined above
PYBIND11_MODULE(neworder, m)
{
  // py::options options;
  // options.disable_function_signatures();
#include "Module_docstr.cpp" 

  // utility/diagnostics
  m.def("version", no::module_version, version_docstr)
   .def("python", no::python_version)
   .def("log", log_obj, log_docstr, "obj"_a)
   .def("run", no::Model::run, run_docstr, "model"_a)
   .def("verbose", no::Environment::verbose, verbose_docstr, "verbose"_a = true)
   .def("checked", no::Environment::checked, checked_docstr, "checked"_a = true);

  // time-related module
  m.attr("time") = py::module("time")
  // TODO move static methods into namespace for consistency?
   .def("distant_past", no::Timeline::distant_past, time_distant_past_docstr)
   .def("far_future", no::Timeline::far_future, time_far_future_docstr)
   .def("never", no::Timeline::never, time_never_docstr)
   .def("isnever", no::Timeline::isnever, time_isnever_docstr, "t"_a) // scalar 
   .def("isnever", no::isnever, time_isnever_a_docstr, "a"_a); // array

  py::class_<no::Timeline>(m, "Timeline")
    .def(py::init<double, double, const std::vector<size_t>&>())
    .def(py::init<>())
    .def_static("null", []() { return no::Timeline(); } ) // calls default ctor (rust workaround, pyo3 doesnt permit multiple ctors)
    .def("index", &no::Timeline::index)
    .def("time", &no::Timeline::time)
    .def("dt", &no::Timeline::dt)
    .def("nsteps", &no::Timeline::nsteps)
    //.def("next", &no::Timeline::next) not exposed 
    .def("at_checkpoint", &no::Timeline::at_checkpoint)
    .def("at_end", &no::Timeline::at_end)
    .def("__repr__", &no::Timeline::repr);

  // Microsimulation (or ABM) model class
  py::class_<no::Model>(m, "Model")
    .def(py::init<no::Timeline&, const py::function&>())
    .def("timeline", &no::Model::timeline, py::return_value_policy::reference)
    .def("mc", &no::Model::mc, py::return_value_policy::reference)
    .def("modify", &no::Model::modify)
    .def("step", &no::Model::step)
    .def("check", &no::Model::check)
    .def("checkpoint", &no::Model::checkpoint);
    // NB the all-important run function is not exposed to python, it can only be executed via the `neworder.run` function

  // MC
  py::class_<no::MonteCarlo>(m, "MonteCarlo")
    .def_static("deterministic_identical_stream", &no::MonteCarlo::deterministic_identical_stream)
    .def_static("deterministic_independent_stream", &no::MonteCarlo::deterministic_independent_stream)
    .def_static("nondeterministic_stream", &no::MonteCarlo::nondeterministic_stream)
    .def("seed", &no::MonteCarlo::seed)  
    .def("reset", &no::MonteCarlo::reset)  
    .def("ustream", &no::MonteCarlo::ustream)
    // explicitly give function type for overloads 
    .def("hazard", py::overload_cast<double, py::ssize_t>(&no::MonteCarlo::hazard), "simulate outcomes from a flat hazard rate")
    .def("hazard", py::overload_cast<const py::array&>(&no::MonteCarlo::hazard), "simulate outcomes from hazard rates")
    .def("stopping", py::overload_cast<double, py::ssize_t>(&no::MonteCarlo::stopping), "simulate stopping times from a flat hazard rate")
    .def("stopping", py::overload_cast<const py::array&>(&no::MonteCarlo::stopping), "simulate stopping times from hazard rates")
    .def("arrivals", &no::MonteCarlo::arrivals)
    .def("first_arrival", &no::MonteCarlo::first_arrival/*, py::arg("minval") = 0.0*/)
    .def("first_arrival", [](no::MonteCarlo& mc, const py::array& lambda_t, double dt, size_t n) { 
        return mc.first_arrival(lambda_t, dt, n, 0.0); 
      })
    .def("next_arrival", &no::MonteCarlo::next_arrival)
    .def("next_arrival", [](no::MonteCarlo& mc, const py::array& startingpoints, const py::array& lambda_t, double dt, bool relative) { 
        return mc.next_arrival(startingpoints, lambda_t, dt, relative, 0.0); 
      })
    .def("next_arrival", [](no::MonteCarlo& mc, const py::array& startingpoints, const py::array& lambda_t, double dt) { 
        return mc.next_arrival(startingpoints, lambda_t, dt, false, 0.0); 
      })
    .def("__repr__", &no::MonteCarlo::repr);
    
    // .def("first_arrival", [](no::MonteCarlo& mc, const py::array& lambda_t, double dt, size_t n) { 
    //   return mc.first_arrival(lambda_t, dt, n, 0.0); 
    // })


  // statistical utils
  m.attr("stats") = py::module("stats", "statistical functions")
    .def("logistic", no::logistic)  // this version defaults x0, k args 
    .def("logistic", [](const py::array& a, double x0) { return no::logistic(a, x0, 1.0); })
    .def("logistic", [](const py::array& a) { return no::logistic(a, 0.0, 1.0); })
    .def("logit", no::logit);
 
  // dataframe manipulation  
  m.attr("dataframe") = py::module("dataframe", "Direct manipulations of dataframes")
    .def("transition", no::df::transition)
    .def("directmod", no::df::directmod);
    //.def("linked_change", no::df::linked_change, py::return_value_policy::take_ownership);

  // MPI submodule
  m.attr("mpi") = py::module("mpi", "Basic MPI environment discovery")
    .def("rank", no::Environment::rank, mpi_rank_docstr)
    .def("size", no::Environment::size, mpi_size_docstr);
    
  no::Environment::init(-1, -1, false, true);

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


