
#include "Environment.h"
#include "Inspect.h"
#include "Module.h"
#include "Log.h"

#include <algorithm>
#include <string>


// This function must be used to init the environment
pycpp::Environment& pycpp::Environment::init(int rank, int size)
{
  // make our rank/size visible to python
  Environment& env = Global::instance<Environment>();

  // Cannot avoid duplicating scalars between C++ and python due to the latter's immutability
  env.m_rank = rank;
  env.m_size = size;

  // TODO consider using MPI terminology
  env.m_self->attr("procid") = env.m_rank;
  env.m_self->attr("nprocs") = env.m_size;

  neworder::log("env: python %%"_s % version());

  return env;
}

void pycpp::Environment::configure()
{
  if (pycpp::has_attr(*m_self, "sync_streams"))
  {
    m_sync_streams = py::extract<bool>(m_self->attr("sync_streams"));
    //neworder::log("sync attr = %%"_s % env.sync_streams());
  }

  // TODO python func to set sequence and reset rng
  if (pycpp::has_attr(*m_self, "sequence"))
  {
    seed(np::from_object(m_self->attr("sequence")));
  } 
}

// syntactic sugar
pycpp::Environment& pycpp::getenv()
{
  return Global::instance<Environment>();
}


int pycpp::Environment::rank() const
{
  return m_rank;
}

// MPI size (1 if serial)
int pycpp::Environment::size() const
{
  return m_size;
}

std::string pycpp::Environment::context(int ctx) const
{
  std::string idstring = "[%% %%-%%/%%] "_s % (ctx == 0 ? "no" : "py") % pycpp::at<int64_t>(sequence(), seq()) % m_rank % m_size;
  return idstring;
}

// Take next stream
bool pycpp::Environment::next()
{
  if (static_cast<size_t>(seq()) == pycpp::size(sequence()) - 1)
    return false;

  neworder::Callback::exec("neworder.seq = neworder.seq + 1")();

  m_prng.seed(compute_seed());
  int64_t seq_val = pycpp::at<int64_t>(sequence(), seq());

  neworder::log("seq: %% sync=%% seed=%%"_s % seq_val % sync_streams() % compute_seed());

  return true;
}

//
bool& pycpp::Environment::sync_streams()
{
  return m_sync_streams;
}

// TODO rename seq_index for clarity 
int pycpp::Environment::seq() const
{
  int s;
  if (pycpp::has_attr(*m_self, "seq"))
  {
    s = py::extract<int>(m_self->attr("seq"));
  }
  else
  {
    throw std::runtime_error("seq not defined");
  }
  np::ndarray a = sequence();

  // if (s<0 || s >= (int)pycpp::size(a))
  // {
  //   throw std::runtime_error("seq out of bounds: %%"_s % s);
  // }
  return s;
}

np::ndarray pycpp::Environment::sequence() const
{
  if (pycpp::has_attr(*m_self, "sequence"))
  {
    return np::from_object(m_self->attr("sequence"));
  }
  else 
  {
    throw("seq not defined");
  }
}


// compute the RNG seed
int64_t pycpp::Environment::compute_seed() const
{
  // ensure stream (in)dependence w.r.t. sequence and MPI rank/sizes
  return 77027473 * pycpp::at<int64_t>(sequence(), seq()) + 19937 * m_size + (m_rank * !m_sync_streams);  
}

// Sets a PRNG sequence (and resets sequence counter)
// TODO rename
void pycpp::Environment::seed(const np::ndarray& seq)
{
  m_self->attr("sequence") = seq;
  m_self->attr("seq") = -1;
  next();
}

std::mt19937& pycpp::Environment::prng()
{
  return m_prng;
}

// Note this does not fully initialise, do not construct directly, use the static init function
pycpp::Environment::Environment() //: m_sequence(pycpp::zero_1d_array<int64_t>(1))
{
  // make the neworder module available in embedded python env
  neworder::import_module();

  // Init python env
  Py_Initialize();

  // init numpy
  np::initialize();

  m_self = new py::object(py::import("neworder"));
  
  // dummy sequence (needs to be read from config.py - which hasnt been loaded yet)
  m_self->attr("sequence") = pycpp::zero_1d_array<int64_t>(1);
  m_self->attr("seq") = 0;
} 

pycpp::Environment::~Environment() 
{
  // Python >=3.6
  // if (Py_FinalizeEx() < 0)
  // {
  //   // report an error...somehow
  // }
  Py_Finalize();
}

// check for errors in the python env: if it returns, there is no error
// copied from: https://wiki.python.org/moin/boost.python/EmbeddingPython
std::string pycpp::Environment::get_error() noexcept
{
  // see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
  std::string message;
  if (PyErr_Occurred())
  {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_NormalizeException(&exc, &val, &tb);

    py::handle<> hexc(exc), hval(py::allow_null(val)), htb(py::allow_null(tb));

    PyErr_Clear();
    if(!hval)
    {
      return py::extract<std::string>(py::str(hexc));
    }
    else
    {
      py::object traceback(py::import("traceback"));
      py::object format_exception(traceback.attr("format_exception"));
      py::object formatted_list(format_exception(hexc,hval,htb));
      py::object formatted(py::str("").join(formatted_list));
      return py::extract<std::string>(formatted);
    } 
  }
  return "unable to determine python error";
}

std::string pycpp::Environment::version()
{
  static std::string version_string;
  // Get and display python version - only do once
  if (version_string.empty())
  {
    py::object sys = py::import("sys");
    version_string = py::extract<std::string>(sys.attr("version"));
    std::replace(version_string.begin(), version_string.end(), '\n', ' ');
  }
  return version_string;
}

