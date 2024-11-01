
#include "Module.h"

#include "Timeline.h"
#include "Model.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "Log.h"

#include "NewOrder.h"

using namespace py::literals;

namespace {

// not visible to (rest of) C++ - use the function declared in Log.h
void log_obj(const py::object& msg)
{
  py::print(no::env::logPrefix[no::env::Context::PY], msg);
}

}

// initially set to invalid values (so that if model is not properly initialised its immediately apparent)
std::atomic_int no::env::rank = -1;
std::atomic_int no::env::size = -1;
std::atomic_bool no::env::verbose = false;
std::atomic_bool no::env::checked = true;
std::atomic_int64_t no::env::uniqueIndex = -1;
// these types are not trivially copyable so can't be atomic
std::string no::env::logPrefix[2];


void init_env(py::object mpi)
{
  int r = 0;
  int s = 1;
  py::object comm = py::none();
  try
  {
    py::module mpi = py::module::import("mpi4py.MPI");
    comm = mpi.attr("COMM_WORLD");
    r = comm.attr("Get_rank")().cast<int>();
    s = comm.attr("Get_size")().cast<int>();
  }
  catch(const py::error_already_set& pyerror)
  {
    // if something other than module not found has occurred, fail
    if (!pyerror.matches(PyExc_ModuleNotFoundError)) throw;
    no::warn("neworder installed in serial mode. If necessary, enable MPI with: pip install neworder[parallel]");
  }

  mpi.attr("COMM") = comm;
  mpi.attr("RANK") = r;
  mpi.attr("SIZE") = s;

  no::env::rank.store(r, std::memory_order_relaxed);
  no::env::size.store(s, std::memory_order_relaxed);
  no::env::uniqueIndex.store(static_cast<int64_t>(no::env::rank), std::memory_order_relaxed);

  // cache log message context for efficiency
  no::env::logPrefix[no::env::Context::CPP] = "[no %%/%%]"s % r % s;
  no::env::logPrefix[no::env::Context::PY] = "[py %%/%%]"s % r % s;
}



// python-visible log function defined above
PYBIND11_MODULE(_neworder_core, m)
{
  // py::options options;
  // options.disable_function_signatures();
#include "Module_docstr.cpp"

  m.doc() = module_docstr;

  // time-related module
  auto time = m.def_submodule("time", time_docstr)
    .def("isnever", no::time::isnever, time_isnever_docstr, "t"_a) // scalar
    .def("isnever", no::time::isnever_a, time_isnever_a_docstr, "t"_a); // array
  time.attr("DISTANT_PAST") = no::time::distant_past();
  time.attr("FAR_FUTURE") = no::time::far_future();
  time.attr("NEVER") = no::time::never();

  // register abstract base class
  py::class_<no::Timeline, no::PyTimeline>(m, "Timeline")
    .def(py::init<>())
    .def_property_readonly("time", &no::Timeline::time, timeline_time_docstr)
    .def_property_readonly("start", &no::Timeline::start, timeline_start_docstr)
    .def_property_readonly("end", &no::Timeline::end, timeline_end_docstr)
    .def_property_readonly("index", &no::Timeline::index, timeline_index_docstr)
    .def_property_readonly("nsteps", &no::Timeline::nsteps, timeline_nsteps_docstr)
    .def_property_readonly("dt", &no::Timeline::dt, timeline_dt_docstr)
    .def_property_readonly("at_end", &no::Timeline::at_end, timeline_at_end_docstr)
    .def("__repr__", &no::Timeline::repr, timeline_repr_docstr)
    ;

  py::class_<no::NoTimeline, no::Timeline>(m, "NoTimeline", notimeline_docstr)
    .def(py::init<>(), notimeline_init_docstr);

  py::class_<no::LinearTimeline, no::Timeline>(m, "LinearTimeline", lineartimeline_docstr)
    .def(py::init<double, double, size_t>(), lineartimeline_init_docstr, "start"_a, "end"_a, "nsteps"_a)
    .def(py::init<double, double>(), lineartimeline_init_open_docstr, "start"_a, "step"_a);

  py::class_<no::NumericTimeline, no::Timeline>(m, "NumericTimeline", numerictimeline_docstr)
    .def(py::init<const std::vector<double>&>(), numerictimeline_init_docstr, "times"_a);

  py::class_<no::CalendarTimeline, no::Timeline>(m, "CalendarTimeline", calendartimeline_docstr)
    .def(py::init<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point, size_t, char>(),
      calendartimeline_init_docstr, "start"_a, "end"_a, "step"_a, "unit"_a)
    .def(py::init<std::chrono::system_clock::time_point, size_t, char>(),
      calendartimeline_init_open_docstr, "start"_a, "step"_a, "unit"_a);

  // MC
  py::class_<no::MonteCarlo>(m, "MonteCarlo", mc_docstr)
    // constructor is NOT exposed to python, can only be created within a model
    .def_static("deterministic_identical_stream", &no::MonteCarlo::deterministic_identical_stream, mc_deterministic_identical_stream_docstr)
    .def_static("deterministic_independent_stream", &no::MonteCarlo::deterministic_independent_stream, mc_deterministic_independent_stream_docstr)
    .def_static("nondeterministic_stream", &no::MonteCarlo::nondeterministic_stream, mc_nondeterministic_stream_docstr)
    .def("init_bitgen", &no::MonteCarlo::init_bitgen, "internal helper function used by as_np")
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
                     "lambda_"_a, "n"_a)
    .def("stopping", py::overload_cast<const py::array_t<double>&>(&no::MonteCarlo::stopping),
                     mc_stopping_a_docstr,
                     "lambda_"_a)
    .def("counts", &no::MonteCarlo::counts,
                   mc_counts_docstr,
                   "lambda_"_a, "dt"_a)
    .def("arrivals", &no::MonteCarlo::arrivals,
                     mc_arrivals_docstr,
                     "lambda_"_a , "dt"_a, "n"_a, "mingap"_a)
    .def("first_arrival", &no::MonteCarlo::first_arrival,
                          mc_first_arrival_docstr,
                          "lambda_"_a, "dt"_a, "n"_a, "minval"_a)
    .def("first_arrival", [](no::MonteCarlo& self, const py::array_t<double>& lambda_t, double dt, size_t n) {
                            return self.first_arrival(lambda_t, dt, n, 0.0);
                          },
                          mc_first_arrival3_docstr,
                          "lambda_"_a, "dt"_a, "n"_a)
    .def("next_arrival", &no::MonteCarlo::next_arrival,
                         mc_next_arrival_docstr,
                         "startingpoints"_a, "lambda_"_a, "dt"_a, "relative"_a, "minsep"_a)
    .def("next_arrival", [](no::MonteCarlo& self, const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt, bool relative) {
                           return self.next_arrival(startingpoints, lambda_t, dt, relative, 0.0);
                         },
                         mc_next_arrival4_docstr,
                         "startingpoints"_a, "lambda_"_a, "dt"_a, "relative"_a)
    .def("next_arrival", [](no::MonteCarlo& self, const py::array_t<double>& startingpoints, const py::array_t<double>& lambda_t, double dt) {
                           return self.next_arrival(startingpoints, lambda_t, dt, false, 0.0);
                         },
                         mc_next_arrival3_docstr,
                         "startingpoints"_a, "lambda_"_a, "dt"_a)
    .def("__repr__", &no::MonteCarlo::repr, mc_repr_docstr);

  // Microsimulation (or ABM) model class
  auto model = py::class_<no::Model, no::PyModel>(m, "Model", model_docstr)
    .def(py::init<no::Timeline&, const py::function&>(), model_init_docstr,"timeline"_a,
       "seeder"_a = py::cpp_function(no::MonteCarlo::deterministic_independent_stream))
    // properties are readonly only in the sense you can't assign to them; you CAN call their mutable methods
    .def_property_readonly("timeline", &no::Model::timeline, model_timeline_docstr)
    .def_property_readonly("mc", &no::Model::mc, model_mc_docstr)
    .def_property_readonly("run_state", &no::Model::runState, model_runstate_docstr)
    .def("modify", &no::Model::modify, model_modify_docstr)
    .def("step", &no::Model::step, model_step_docstr)
    .def("check", &no::Model::check, model_check_docstr)
    .def("finalise", &no::Model::finalise, model_finalise_docstr)
    .def("halt", &no::Model::halt, model_halt_docstr);
    // NB the all-important run function is not exposed to python, it can only be executed via the `neworder.run` function

  py::enum_<no::Model::RunState>(model, "RunState")
  .value("NOT_STARTED", no::Model::NOT_STARTED)
  .value("RUNNING", no::Model::RUNNING)
  .value("HALTED", no::Model::HALTED)
  .value("COMPLETED", no::Model::COMPLETED)
  .export_values();

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

  // model control plus utility/diagnostics
  m.def("log", log_obj, log_docstr, "obj"_a)
   .def("run", no::Model::run, run_docstr, "model"_a)
   .def("verbose", [](bool v = true) { no::env::verbose.store(v, std::memory_order_relaxed); }, verbose_docstr, "verbose"_a = true)
   .def("checked", [](bool c = true) { no::env::checked.store(c, std::memory_order_relaxed); }, checked_docstr, "checked"_a = true);

  // MPI submodule
  auto mpi = m.def_submodule("mpi", mpi_docstr);
  init_env(mpi);
}

