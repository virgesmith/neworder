
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

  // dummy sequence
  env.m_self->attr("sequence") = pycpp::zero_1d_array<int64_t>(1);
  env.m_self->attr("seq") = 0;

  neworder::log("embedded python version: %%"_s % version());
  neworder::log("env init");

  return env;
}

// syntactic sugar
pycpp::Environment& pycpp::Environment::get()
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
  np::ndarray sequence = np::from_object(m_self->attr("sequence"));
  std::string idstring = "[%% %%-%%/%%] "_s % (ctx == 0 ? "no" : "py") % pycpp::at<int64_t>(sequence, m_seqno) % m_rank % m_size;
  return idstring;
}

// Take next stream
bool pycpp::Environment::next()
{
  np::ndarray sequence = np::from_object(m_self->attr("sequence"));
  if (m_seqno == pycpp::size(sequence) - 1)
    return false;
  ++m_seqno;

  // ensure stream indepence w.r.t. sequence and MPI rank/size
  m_prng->seed(77027473 * pycpp::at<int64_t>(sequence, m_seqno) + 19937 * m_size + m_rank);
  m_self->attr("seq") = pycpp::at<int64_t>(sequence, m_seqno);

  return true;
}

  // compute the RNG seed
int64_t pycpp::Environment::compute_seed() const
{
  np::ndarray sequence = np::from_object(m_self->attr("sequence"));

  // ensure stream (in)dependence w.r.t. sequence and MPI rank/sizes
  return 77027473 * pycpp::at<int64_t>(sequence, m_seqno) + 19937 * m_size + m_rank * !m_sync_streams;  
}

// Sets a PRNG sequence (and resets sequence counter)
void pycpp::Environment::seed(const np::ndarray& seq)
{
  m_self->attr("sequence") = seq;
  m_seqno = -1;
  next();
}

std::mt19937& pycpp::Environment::prng()
{
  // move to check?
  if (!m_prng)
    throw std::runtime_error("mt not init");
  return *m_prng;
}

// Note this does not fully initialise, do not construct directly, use the static init function
pycpp::Environment::Environment()
{
  // make the neworder module available in embedded python env
  neworder::import_module();

  // Init python env
  Py_Initialize();

  // init numpy
  np::initialize();

  m_self = new py::object(py::import("neworder"));
  

  // Init rng (unseeded for now)
  // TODO this doesnt need to be a unique_ptr now?
  m_prng = std::make_unique<std::mt19937>();
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
    PyObject *exc,*val,*tb;
    PyErr_Fetch(&exc,&val,&tb);
    PyErr_NormalizeException(&exc,&val,&tb);

    py::handle<> hexc(exc), hval(py::allow_null(val)), htb(py::allow_null(tb));
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
    version_string = py::extract<std::string>(sys.attr("version"))();
    std::replace(version_string.begin(), version_string.end(), '\n', ' ');
  }
  return version_string;
}

