
#include "Module.h"

#include "Version.h"
#include "Timeline.h"
#include "Model.h"
#include "Environment.h"
#include "Inspect.h"
#include "MonteCarlo.h"
#include "NPArray.h"
#include "DataFrame.h"
#include "MPIComms.h"

#include "NewOrder.h"

#include <iostream>

namespace {

// not visible to (rest of) C++ - use the function declared in Log.h
void log_obj(const py::object& msg)
{
  std::cout << no::getenv().context(no::Environment::PY) << msg << std::endl;
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
  py::dict locals = py::module::import("__main__").attr("__dict__");
  //locals/*["neworder"]*/ = py::module::import("neworder").attr("__dict__");
  locals["neworder"] = py::module::import("neworder");
  py::dict kwargs;
  kwargs["banner"] = py::str("[starting neworder debug shell]");
  kwargs["exitmsg"] = py::str("[exiting neworder debug shell]");
  kwargs["local"] = locals; 
  //kwargs["local"] = py::handle(PyObject_Dir(py::module::import("neworder").ptr()));
  /* py::object interpreter = */py::module::import("code").attr("interact")(/**py::tuple(),*/ **kwargs);
}


// // TODO move? (this is called from run.cpp but not exposed to python) 
// void no::set_timeline(const py::tuple& spec) 
// {
//   // Format is: (start, end, [checkpoints])
//   if (py::len(spec) != 3) {
//     throw std::runtime_error("invalid timeline specification, should be (start, end, [checkpoints])");
//   }
//   double start = spec[0].cast<double>();
//   double end = spec[1].cast<double>();
//   py::list cp = py::list(spec[2]);
//   size_t n = py::len(cp);
//   std::vector<size_t> checkpoint_times(n);
//   for (size_t i = 0; i < n; ++i)
//   {
//     checkpoint_times[i] = cp[i].cast<size_t>();
//   }
//   getenv().timeline() = Timeline(start, end, checkpoint_times);
// }


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
  m.def("isnever", no::isnever); // array

  // statistical utils
      // .def("first_arrival", [](no::MonteCarlo& mc, const py::array& lambda_t, double dt, size_t n) { 
      //   return mc.first_arrival(lambda_t, dt, n, 0.0); 
      // })

  // this version defaults x0, k args 
  m.def("logistic", no::logistic);
  m.def("logistic", [](const py::array& a, double x0) { return no::logistic(a, x0, 1.0); });
  m.def("logistic", [](const py::array& a) { return no::logistic(a, 0.0, 1.0); });
  m.def("logit", no::logit);

  py::class_<no::Timeline>(m, "Timeline")
    .def(py::init<double, double, const std::vector<size_t>&>())
    .def(py::init<>())
    .def_static("null", []() { return no::Timeline(); } ) // calls default ctor (rust workaround, pyo3 doesnt permit multiple ctors)
    .def("index", &no::Timeline::index)
    .def("time", &no::Timeline::time)
    .def("dt", &no::Timeline::dt)
    .def("nsteps", &no::Timeline::nsteps)
    .def("next", &no::Timeline::next)
    .def("at_end", &no::Timeline::at_end)
    .def("__repr__", [](const no::Timeline& tl) {
        return "<neworder.Timeline start=%% end=%% checkpoints=%% index=%%>"_s 
          % tl.start() % tl.end() % tl.checkpoints() % tl.index();
      }
    );

  // Microsimulation (or ABM) model class
  py::class_<no::Model>(m, "Model")
    .def(py::init<no::Timeline&>())
    .def("timeline", &no::Model::timeline, py::return_value_policy::reference)
    .def("modify", &no::Model::modify)
    .def("transition", &no::Model::transition)
    .def("check", &no::Model::check)
    .def("checkpoint", &no::Model::checkpoint);
    // NB the all-important run function is not exposed to python

  // MC
  py::class_<no::MonteCarlo>(m, "MonteCarlo")
  //.def(py::init<uint32_t>()); // no ctor visible to python  
    .def("indep", &no::MonteCarlo::indep)
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
    .def("__repr__", [](const no::MonteCarlo& mc) {
        return "<neworder.MonteCarlo indep=%% seed=%%>"_s 
          % mc.indep() % mc.seed();
      });

  // working on pandas df manipulation  
  m.def("transition", no::df::transition);
  m.def("directmod", no::df::directmod);
  //m.def("linked_change", no::df::linked_change, py::return_value_policy::take_ownership);

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

  // // Deferred eval/exec of Python code
  // py::class_<no::Callback>(m, "Callback"/*, py::no_init*/)
  //   .def("__call__", &no::Callback::operator())
  //   .def("__str__", &no::Callback::code)
  //   ;

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


