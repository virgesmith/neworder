
#include "Module.h"

#include "Timeline.h"
#include "Model.h"
#include "Environment.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "Log.h"
#include "Error.h"

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

// jump through hoops as msvc seems to strip quotes from defines, so need to add them here
#define STR2(x) #x
#define STR(x) STR2(x)

const char* no::module_version()
{
  return STR(NEWORDER_VERSION);
}

// std::string no::python_version()
// {
//   return no::getenv().python_version();
// }

// python-visible log function defined above
PYBIND11_MODULE(neworder, m)
{
  // py::options options;
  // options.disable_function_signatures();
#include "Module_docstr.cpp" 

  // utility/diagnostics
  m.def("version", no::module_version, version_docstr)
   .def("python", [](){ no::getenv().python_version(); })
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
   .def("isnever", no::isnever, time_isnever_a_docstr, "y"_a); // array

  py::class_<no::Timeline>(m, "Timeline", "Timestepping functionality")
    .def(py::init<double, double, const std::vector<size_t>&>(), empty_docstr, "start"_a, "end"_a, "checkpoints"_a)
    //.def(py::init<>())
    .def_static("null", []() { return no::Timeline(); }, empty_docstr) // calls default ctor (rust workaround, pyo3 doesnt permit multiple ctors)
    // Only const accessors exposed to python
    .def("start", &no::Timeline::start, empty_docstr)
    .def("end", &no::Timeline::end, empty_docstr)
    .def("index", &no::Timeline::index, empty_docstr)
    .def("time", &no::Timeline::time, empty_docstr)
    .def("dt", &no::Timeline::dt, empty_docstr)
    .def("nsteps", &no::Timeline::nsteps, empty_docstr)
    //.def("next", &no::Timeline::next) not exposed 
    .def("at_checkpoint", &no::Timeline::at_checkpoint, empty_docstr)
    .def("at_end", &no::Timeline::at_end, empty_docstr)
    .def("__repr__", &no::Timeline::repr, empty_docstr);

  // Microsimulation (or ABM) model class
  py::class_<no::Model>(m, "Model", "The base model class from which all neworder models should be subclassed")
    .def(py::init<no::Timeline&, const py::function&>(), 
         model_init_docstr, 
         "timeline"_a, "seeder"_a)
    .def("timeline", &no::Model::timeline, py::return_value_policy::reference, model_timeline_docstr)
    .def("mc", &no::Model::mc, py::return_value_policy::reference, model_mc_docstr)
    .def("modify", &no::Model::modify, model_modify_docstr, "r"_a)
    .def("step", &no::Model::step, model_step_docstr)
    .def("check", &no::Model::check, model_check_docstr)
    .def("checkpoint", &no::Model::checkpoint, model_checkpoint_docstr);
    // NB the all-important run function is not exposed to python, it can only be executed via the `neworder.run` function

  // MC
  py::class_<no::MonteCarlo>(m, "MonteCarlo", "The model's Monte-Carlo engine")
    // constructor is NOT exposed to python, can only be created withing a model
    .def_static("deterministic_identical_stream", &no::MonteCarlo::deterministic_identical_stream, empty_docstr, "r"_a)
    .def_static("deterministic_independent_stream", &no::MonteCarlo::deterministic_independent_stream, empty_docstr, "r"_a)
    .def_static("nondeterministic_stream", &no::MonteCarlo::nondeterministic_stream, empty_docstr, "r"_a)
    .def("seed", &no::MonteCarlo::seed, empty_docstr)
    .def("reset", &no::MonteCarlo::reset, empty_docstr)  
    .def("ustream", &no::MonteCarlo::ustream, empty_docstr, "n"_a)
    // explicitly give function type for overloads 
    .def("hazard", py::overload_cast<double, py::ssize_t>(&no::MonteCarlo::hazard), 
                   empty_docstr,
                   "p"_a, "n"_a)
    .def("hazard", py::overload_cast<const py::array_t<double>&>(&no::MonteCarlo::hazard), 
                   empty_docstr,
                   "p"_a)
    .def("stopping", py::overload_cast<double, py::ssize_t>(&no::MonteCarlo::stopping), 
                     empty_docstr,
                     "p"_a, "n"_a)
    .def("stopping", py::overload_cast<const py::array_t<double>&>(&no::MonteCarlo::stopping), 
                     empty_docstr,
                     "p"_a)
    .def("arrivals", &no::MonteCarlo::arrivals, 
                     empty_docstr,
                     "lambda"_a , "dt"_a, "mingap"_a, "n"_a)
    .def("first_arrival", &no::MonteCarlo::first_arrival, 
                          empty_docstr,
                          "lambda"_a, "dt"_a, "n"_a, "minval"_a)
    .def("first_arrival", [](no::MonteCarlo& self, const py::array_t<double>& lambda_t, double dt, size_t n) { 
                            return self.first_arrival(lambda_t, dt, n, 0.0); 
                          }, 
                          empty_docstr,
                          "lambda"_a, "dt"_a, "n"_a)
    .def("next_arrival", &no::MonteCarlo::next_arrival, 
                         empty_docstr,
                         "startingpoints"_a, "lambda"_a, "dt"_a, "relative"_a, "minsep"_a)
    .def("next_arrival", [](no::MonteCarlo& self, const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt, bool relative) { 
                           return self.next_arrival(startingpoints, lambda_t, dt, relative, 0.0); 
                         }, 
                         empty_docstr,
                         "startingpoints"_a, "lambda"_a, "dt"_a, "relative"_a)
    .def("next_arrival", [](no::MonteCarlo& self, const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt) { 
                           return self.next_arrival(startingpoints, lambda_t, dt, false, 0.0); 
                         }, 
                         empty_docstr,
                         "startingpoints"_a, "lambda"_a, "dt"_a)
    .def("__repr__", &no::MonteCarlo::repr, empty_docstr);
    
    // .def("first_arrival", [](no::MonteCarlo& mc, const py::array_t<double>& lambda_t, double dt, size_t n) { 
    //   return mc.first_arrival(lambda_t, dt, n, 0.0); 
    // })

  // statistical utils
  m.attr("stats") = py::module("stats", "statistical functions")
    .def("logistic", no::logistic, 
                     stats_logistic_docstr, 
                     "x"_a, "x0"_a, "k"_a)   
    .def("logistic", [](const py::array_t<double>& a, double k) { return no::logistic(a, 0.0, k); }, 
                     stats_logistic_docstr_2, 
                     "x"_a, "k"_a)
    .def("logistic", [](const py::array_t<double>& a) { return no::logistic(a, 0.0, 1.0); }, 
                     stats_logistic_docstr_1,
                     "x"_a)
    .def("logit",   no::logit, 
                    stats_logit_docstr, 
                    "x"_a);
 
  // dataframe manipulation  
  m.attr("df") = py::module("df", "Direct manipulations of dataframes")
    .def("transition", no::df::transition, df_transition_docstr, "model"_a, "categories"_a, "transition_matrix"_a, "df"_a, "colname"_a)
    .def("testfunc", no::df::testfunc, df_testfunc_docstr, "model"_a, "df"_a, "colname"_a);
    //.def("linked_change", no::df::linked_change, py::return_value_policy::take_ownership);

  // MPI submodule
  m.attr("mpi") = py::module("mpi", "Basic MPI environment discovery")
    .def("rank", no::Environment::rank, mpi_rank_docstr)
    .def("size", no::Environment::size, mpi_size_docstr);
    
  no::Environment::init(-1, -1, false, true);

  // Map custom C++ exceptions to python ones
  py::register_exception_translator(no::exception_translator);
}


