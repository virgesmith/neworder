
#include "Module.h"

#include "Timeline.h"
#include "Model.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "Log.h"
#include "Error.h"

#include "NewOrder.h"

using namespace py::literals;

namespace {

// not visible to (rest of) C++ - use the function declared in Log.h
void log_obj(const py::object& msg)
{
  py::print(no::env::logPrefix[no::env::Context::PY], msg);
}

}

// jump through hoops as msvc seems to strip quotes from defines, so need to add them here
#define STR2(x) #x
#define STR(x) STR2(x)

const char* no::module_version()
{
  return STR(NEWORDER_VERSION);
}


void init_env()
{
  try
  {
    py::module mpi = py::module::import("mpi4py.MPI");
    py::object comm = mpi.attr("COMM_WORLD");
    no::env::rank.store(comm.attr("Get_rank")().cast<int>(), std::memory_order_relaxed);
    no::env::size.store(comm.attr("Get_size")().cast<int>(), std::memory_order_relaxed);
  }
  catch(const py::error_already_set& pyerror)
  {
    // if something other than module not found has occurred, fail
    if (!pyerror.matches(PyExc_ModuleNotFoundError)) throw;
    no::warn("mpi4py module not found, assuming serial mode");
    no::env::rank.store(0, std::memory_order_relaxed);
    no::env::size.store(1, std::memory_order_relaxed);
  }

  no::env::uniqueIndex.store(static_cast<int64_t>(no::env::rank), std::memory_order_relaxed);

  // cache log message context for efficiency
  int r = no::env::rank.load(std::memory_order_relaxed);
  int s = no::env::size.load(std::memory_order_relaxed);
  no::env::logPrefix[no::env::Context::CPP] = "[no %%/%%]"s % r % s;
  no::env::logPrefix[no::env::Context::PY] = "[py %%/%%]"s % r % s;
}


// python-visible log function defined above
PYBIND11_MODULE(neworder, m)
{
  // py::options options;
  // options.disable_function_signatures();
#include "Module_docstr.cpp"

  m.doc() = module_docstr;

  // model control plus utility/diagnostics

  m.def("version", no::module_version, version_docstr)
   .def("log", log_obj, log_docstr, "obj"_a)
   .def("run", no::Model::run, run_docstr, "model"_a)
   .def("verbose", [](bool v = true) { no::env::verbose.store(v, std::memory_order_relaxed); }, verbose_docstr, "verbose"_a = true)
   .def("checked", [](bool c = true) { no::env::checked.store(c, std::memory_order_relaxed); }, checked_docstr, "checked"_a = true);

  // time-related module

  m.def_submodule("time", time_docstr)
    .def("distant_past", no::time::distant_past, time_distant_past_docstr)
    .def("far_future", no::time::far_future, time_far_future_docstr)
    .def("never", no::time::never, time_never_docstr)
    .def("isnever", no::time::isnever, time_isnever_docstr, "t"_a) // scalar
    .def("isnever", no::time::isnever_a, time_isnever_a_docstr, "t"_a); // array

  py::class_<no::NoTimeline>(m, "NoTimeline", notimeline_docstr)
    .def(py::init<>(), notimeline_init_docstr)
    .def("start", &no::NoTimeline::start, timeline_start_docstr)
    .def("end", &no::NoTimeline::end, timeline_end_docstr)
    .def("index", &no::NoTimeline::index, timeline_index_docstr)
    .def("time", &no::NoTimeline::time, timeline_time_docstr)
    .def("dt", &no::NoTimeline::dt, timeline_dt_docstr)
    .def("nsteps", &no::NoTimeline::nsteps, timeline_nsteps_docstr)
    .def("at_end", &no::NoTimeline::at_end, timeline_at_end_docstr)
    .def("__repr__", &no::NoTimeline::repr, timeline_repr_docstr);

  py::class_<no::LinearTimeline>(m, "LinearTimeline", lineartimeline_docstr)
    .def(py::init<double, double, size_t>(), lineartimeline_init_docstr, "start"_a, "end"_a, "nsteps"_a)
    .def(py::init<double, double>(), lineartimeline_init_open_docstr, "start"_a, "step"_a)
    .def("start", &no::LinearTimeline::start, timeline_start_docstr)
    .def("end", &no::LinearTimeline::end, timeline_end_docstr)
    .def("index", &no::LinearTimeline::index, timeline_index_docstr)
    .def("time", &no::LinearTimeline::time, timeline_time_docstr)
    .def("dt", &no::LinearTimeline::dt, timeline_dt_docstr)
    .def("nsteps", &no::LinearTimeline::nsteps, timeline_nsteps_docstr)
    .def("at_end", &no::LinearTimeline::at_end, timeline_at_end_docstr)
    .def("__repr__", &no::LinearTimeline::repr, timeline_repr_docstr);

  py::class_<no::NumericTimeline>(m, "NumericTimeline", numerictimeline_docstr)
    .def(py::init<const std::vector<double>&>(), numerictimeline_init_docstr, "times"_a)
    .def("start", &no::NumericTimeline::start, timeline_start_docstr)
    .def("end", &no::NumericTimeline::end, timeline_end_docstr)
    .def("index", &no::NumericTimeline::index, timeline_index_docstr)
    .def("time", &no::NumericTimeline::time, timeline_time_docstr)
    .def("dt", &no::NumericTimeline::dt, timeline_dt_docstr)
    .def("nsteps", &no::NumericTimeline::nsteps, timeline_nsteps_docstr)
    .def("at_end", &no::NumericTimeline::at_end, timeline_at_end_docstr)
    .def("__repr__", &no::NumericTimeline::repr, timeline_repr_docstr);

  py::class_<no::CalendarTimeline>(m, "CalendarTimeline", calendartimeline_docstr)
    .def(py::init<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point, size_t, char>(),
      calendartimeline_init_docstr, "start"_a, "end"_a, "step"_a, "unit"_a)
    .def(py::init<std::chrono::system_clock::time_point, size_t, char>(),
      calendartimeline_init_open_docstr, "start"_a, "step"_a, "unit"_a)
    .def("start", &no::CalendarTimeline::start, timeline_start_docstr)
    .def("end", &no::CalendarTimeline::end, timeline_end_docstr)
    .def("index", &no::CalendarTimeline::index, timeline_index_docstr)
    .def("time", &no::CalendarTimeline::time, timeline_time_docstr)
    .def("dt", &no::CalendarTimeline::dt, timeline_dt_docstr)
    .def("nsteps", &no::CalendarTimeline::nsteps, timeline_nsteps_docstr)
    .def("at_end", &no::CalendarTimeline::at_end, timeline_at_end_docstr)
    .def("__repr__", &no::CalendarTimeline::repr, timeline_repr_docstr);

  // Microsimulation (or ABM) model class
  py::class_<no::Model>(m, "Model", model_docstr)
    .def(py::init([](no::NoTimeline& t, const py::function& s) { return no::Model(std::make_unique<no::NoTimeline>(t), s); }), model_init_notimeline_docstr,"timeline"_a, "seeder"_a)
    .def(py::init([](no::LinearTimeline& t, const py::function& s) { return no::Model(std::make_unique<no::LinearTimeline>(t), s); }), model_init_lineartimeline_docstr,"timeline"_a, "seeder"_a)
    .def(py::init([](no::NumericTimeline& t, const py::function& s) { return no::Model(std::make_unique<no::NumericTimeline>(t), s); }), model_init_numerictimeline_docstr,"timeline"_a, "seeder"_a)
    .def(py::init([](no::CalendarTimeline& t, const py::function& s) { return no::Model(std::make_unique<no::CalendarTimeline>(t), s); }), model_init_calendartimelime_docstr,"timeline"_a, "seeder"_a)
    .def("timeline", &no::Model::timeline, py::return_value_policy::reference, model_timeline_docstr)
    .def("mc", &no::Model::mc, py::return_value_policy::reference, model_mc_docstr)
    .def("modify", &no::Model::modify, model_modify_docstr, "r"_a)
    .def("step", &no::Model::step, model_step_docstr)
    .def("check", &no::Model::check, model_check_docstr)
    .def("finalise", &no::Model::finalise, model_finalise_docstr)
    .def("halt", &no::Model::halt, model_halt_docstr);
    // NB the all-important run function is not exposed to python, it can only be executed via the `neworder.run` function

  // MC
  py::class_<no::MonteCarlo>(m, "MonteCarlo", mc_docstr)
    // constructor is NOT exposed to python, can only be created withing a model
    .def_static("deterministic_identical_stream", &no::MonteCarlo::deterministic_identical_stream, mc_deterministic_identical_stream_docstr, "r"_a)
    .def_static("deterministic_independent_stream", &no::MonteCarlo::deterministic_independent_stream, mc_deterministic_independent_stream_docstr, "r"_a)
    .def_static("nondeterministic_stream", &no::MonteCarlo::nondeterministic_stream, mc_nondeterministic_stream_docstr, "r"_a)
    .def("seed", &no::MonteCarlo::seed, mc_seed_docstr)
    .def("reset", &no::MonteCarlo::reset, mc_reset_docstr)
    .def("state", &no::MonteCarlo::state, mc_state_docstr)
    .def("raw", &no::MonteCarlo::raw, mc_raw_docstr)
    .def("ustream", &no::MonteCarlo::ustream, mc_ustream_docstr, "n"_a)
    .def("sample", &no::MonteCarlo::sample, mc_sample_docstr, "n"_a, "cat_weights"_a)
    // explicitly give function type for overloads
    .def("hazard", py::overload_cast<double, py::ssize_t>(&no::MonteCarlo::hazard),
                   mc_hazard_docstr,
                   "p"_a, "n"_a)
    .def("hazard", py::overload_cast<const py::array_t<double>&>(&no::MonteCarlo::hazard),
                   mc_hazard_a_docstr,
                   "p"_a)
    .def("stopping", py::overload_cast<double, py::ssize_t>(&no::MonteCarlo::stopping),
                     mc_stopping_docstr,
                     "lambda"_a, "n"_a)
    .def("stopping", py::overload_cast<const py::array_t<double>&>(&no::MonteCarlo::stopping),
                     mc_stopping_a_docstr,
                     "lambda"_a)
    .def("counts", &no::MonteCarlo::counts,
                   mc_counts_docstr,
                   "lambda"_a, "dt"_a)
    .def("arrivals", &no::MonteCarlo::arrivals,
                     mc_arrivals_docstr,
                     "lambda"_a , "dt"_a, "n"_a, "mingap"_a)
    .def("first_arrival", &no::MonteCarlo::first_arrival,
                          mc_first_arrival_docstr,
                          "lambda"_a, "dt"_a, "n"_a, "minval"_a)
    .def("first_arrival", [](no::MonteCarlo& self, const py::array_t<double>& lambda_t, double dt, size_t n) {
                            return self.first_arrival(lambda_t, dt, n, 0.0);
                          },
                          mc_first_arrival3_docstr,
                          "lambda"_a, "dt"_a, "n"_a)
    .def("next_arrival", &no::MonteCarlo::next_arrival,
                         mc_next_arrival_docstr,
                         "startingpoints"_a, "lambda"_a, "dt"_a, "relative"_a, "minsep"_a)
    .def("next_arrival", [](no::MonteCarlo& self, const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt, bool relative) {
                           return self.next_arrival(startingpoints, lambda_t, dt, relative, 0.0);
                         },
                         mc_next_arrival4_docstr,
                         "startingpoints"_a, "lambda"_a, "dt"_a, "relative"_a)
    .def("next_arrival", [](no::MonteCarlo& self, const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt) {
                           return self.next_arrival(startingpoints, lambda_t, dt, false, 0.0);
                         },
                         mc_next_arrival3_docstr,
                         "startingpoints"_a, "lambda"_a, "dt"_a)
    .def("__repr__", &no::MonteCarlo::repr, mc_repr_docstr);

    // .def("first_arrival", [](no::MonteCarlo& mc, const py::array_t<double>& lambda_t, double dt, size_t n) {
    //   return mc.first_arrival(lambda_t, dt, n, 0.0);
    // })

  // statistical utils
  m.def_submodule("stats", stats_docstr)
    .def("logistic", no::logistic,
                     stats_logistic_docstr,
                     "x"_a, "x0"_a, "k"_a)
    .def("logistic", [](const py::array_t<double>& a, double k) { return no::logistic(a, 0.0, k); },
                     stats_logistic_docstr_2,
                     "x"_a, "k"_a)
    .def("logistic", [](const py::array_t<double>& a) { return no::logistic(a, 0.0, 1.0); },
                     stats_logistic_docstr_1,
                     "x"_a)
    .def("logit",    no::logit,
                     stats_logit_docstr,
                     "x"_a);

  // dataframe manipulation
  m.def_submodule("df", df_docstr)
    .def("unique_index", no::df::unique_index, df_unique_index_docstr, "n"_a)
    .def("transition", no::df::transition, df_transition_docstr, "model"_a, "categories"_a, "transition_matrix"_a, "df"_a, "colname"_a)
    .def("testfunc", no::df::testfunc, df_testfunc_docstr, "model"_a, "df"_a, "colname"_a);
    //.def("linked_change", no::df::linked_change, py::return_value_policy::take_ownership);

  // MPI submodule
  m.def_submodule("mpi", mpi_docstr)
    .def("rank", []() { return no::env::rank.load(std::memory_order_relaxed); }, mpi_rank_docstr)
    .def("size", []() { return no::env::size.load(std::memory_order_relaxed); }, mpi_size_docstr);

  // Map custom C++ exceptions to python ones
  py::register_exception_translator(no::exception_translator);

  init_env();
}

// initially set to invalid values (so that if model is not properly initialised its immediately apparent)
std::atomic_int no::env::rank = -1;
std::atomic_int no::env::size = -1;
std::atomic_bool no::env::verbose = false;
std::atomic_bool no::env::checked = true;
std::atomic_bool no::env::halt = false;
std::atomic_int64_t no::env::uniqueIndex = -1;
// strings are not trivially copyable so can't be atomic
std::string no::env::logPrefix[2];
