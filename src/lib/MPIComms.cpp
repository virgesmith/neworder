#include "MPIComms.h"

#ifdef NEWORDER_MPI

#include <mpi.h>

namespace {

// Buffer for MPI send/recv
class Buffer
{
public:
  typedef char value_type;

  // borrowed
  Buffer(char* b, int n) : m_owned(false), m_buf(b), m_size(n) { }
  // owned
  explicit Buffer(int n) 
  { 
    alloc(n); 
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer()
  {
    free();
  }

  void alloc(int n)
  {
    free();
    m_buf = new char[n];
    m_owned = true;
    m_size = n;
  }

  void free()
  {
    if (m_owned)
      delete[] m_buf;
  }

  bool owned() const 
  {
    return m_owned;
  }

  char* const buf() const 
  {
    return m_buf;
  }

  size_t size() const
  {
    return m_size;
  }

private:
  bool m_owned;
  char* m_buf;
  int m_size;
};

}

#ifdef NEWORDER_MPI
// these specialisation must be in global namespace (as per template)
template<>
struct mpi_type_trait<Buffer>
{
  static constexpr const auto type = MPI_CHAR;
};

template<>
struct mpi_type_trait<std::string>
{
  static constexpr const auto type = MPI_CHAR;
};
#endif

namespace no { namespace mpi {

template<>
void send(const std::string& data, int process)
{
  // no::log("send length %%"_s % size);
  MPI_Send(data.data(), data.size(), mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD);
  // no::log("send %%"_s % data.substr(40));
}

template<>
void send(const Buffer& data, int process)
{
  //MPI_Send(&data.size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
  //no::log("buf send length %%"_s % data.size);
  MPI_Send(data.buf(), data.size(), mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD);
  // no::log("send %%"_s % data.substr(40));
}

template<>
void receive(std::string& data, int process)
{
  // Probe for an incoming message from process 
  MPI_Status status;
  MPI_Probe(process, 0, MPI_COMM_WORLD, &status);

  // When probe returns, the status object has the size and other attributes of the incoming message. Get the message size
  int size;
  MPI_Get_count(&status, mpi_type_trait<std::string>::type, &size);
  no::log("str recv length %%"_s % size);

  data.resize(size);
  MPI_Recv(&data[0], size, mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void receive(Buffer& data, int process)
{
  // Probe for an incoming message from process
  MPI_Status status;
  MPI_Probe(process, 0, MPI_COMM_WORLD, &status);

  // When probe returns, the status object has the size and other attributes of the incoming message. Get the message size
  int size;
  MPI_Get_count(&status, mpi_type_trait<Buffer>::type, &size);
  //no::log("buf recv length %%"_s % size);

  data.alloc(size);
  MPI_Recv(data.buf(), data.size(), mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void broadcast(Buffer& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Bcast(data.buf(), data.size(), MPI_CHAR, process, MPI_COMM_WORLD);
#endif
}

template<>
void broadcast(std::string& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Bcast(&data[0], data.size(), mpi_type_trait<std::string>::type, process, MPI_COMM_WORLD);
#endif
}

}}

#endif


// to the next rank 
void no::mpi::send_obj(const py::object& o, int rank)
{
#ifdef NEWORDER_MPI
  py::object pickle = py::module::import("pickle");
  py::object serialised = pickle.attr("dumps")(o);

  //you can also write checks here for length, verify the 
  //buffer is memory-contiguous, etc.
  Buffer b(PyBytes_AsString(serialised.ptr()), (int)PyBytes_Size(serialised.ptr()));

  // something along the lines of (see https://docs.python.org/3.6/c-api/buffer.html)
  //std::string s(PyBytes_AsString(serialised.ptr()), py::len(serialised)); 
  //no::log("sending %% (len %%) to 1"_s % s % s.size());
  send(b, rank);
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

py::object no::mpi::receive_obj(int rank)
{
#ifdef NEWORDER_MPI
  Buffer b(nullptr, 0);
  receive(b, rank); // b is alloc'd in here

  py::object pickle = py::module::import("pickle");

  py::object o = pickle.attr("loads")(py::handle(PyBytes_FromStringAndSize(b.buf(), b.size())));
  //no::log("got %% from %%"_s % s % rank);
  return o;
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

void no::mpi::send_csv(const py::object& df, int rank)
{
#ifdef NEWORDER_MPI
  py::object io = py::module::import("io");
  py::object buf = io.attr("StringIO")();
  // kwargs broken?
  // py::dict kwargs;
  // kwargs["index"] = false;
  // to_csv(path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True...
  df.attr("to_csv")(buf); //, ", ", "", py::object(), py::object(), true, false); 
  std::string csvbuf = buf.attr("getvalue")().cast<std::string>();
  send(csvbuf, rank);
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

py::object no::mpi::receive_csv(int rank)
{
#ifdef NEWORDER_MPI
  std::string buf;
  receive(buf, rank);

  py::object io = py::module::import("io");
  py::object pd = py::module::import("pandas");
  py::object pybuf = io.attr("StringIO")(buf);
  py::object df = pd.attr("read_csv")(pybuf);
  // temp workaround for not being able to pass to_csv index=False arg
  df = df.attr("drop")("Unnamed: 0", 1);
  return df;
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

// Broadcast object from rank to all other procs
// Have to return by value as (some) python objects are immutable 
py::object no::mpi::broadcast_obj(py::object& o, int rank)
{
#ifdef NEWORDER_MPI
  py::object pickle = py::module::import("pickle");
  py::object serialised = pickle.attr("dumps")(o);
  Buffer b(PyBytes_AsString(serialised.ptr()), (int)PyBytes_Size(serialised.ptr()));
  broadcast(b, rank);
  return pickle.attr("loads")(py::handle(PyBytes_FromStringAndSize(b.buf(), b.size())));
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

py::array no::mpi::gather_array(double x, int rank)
{
#ifdef NEWORDER_MPI
  no::Environment& env = no::getenv();
  py::array ret = no::empty_array<double>({rank == env.rank() ? (size_t)env.size() : 0});
  double* p = (rank == env.rank()) ? no::begin<double>(ret) : nullptr;
  MPI_Gather(&x, 1, mpi_type_trait<double>::type, p, 1, mpi_type_trait<double>::type, rank, MPI_COMM_WORLD);
  return ret;
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

double no::mpi::scatter_array(py::array x, int rank)
{
#ifdef NEWORDER_MPI
  no::Environment& env = no::getenv();
  // If rank=process, return the array, otherwise return an empty array
  double dest;
  double* p = nullptr;
  if (env.rank() == rank)
  {
    if (x.size() < (size_t)env.size())
      throw std::runtime_error("scatter array size %% is smaller than MPI size (%%)"_s % x.size() % env.size());
    p = no::begin<double>(x);
  }
  MPI_Scatter(p, 1, mpi_type_trait<double>::type, &dest, 1, mpi_type_trait<double>::type, rank, MPI_COMM_WORLD);
  return dest;
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

py::array no::mpi::allgather_array(py::array source_dest)
{
#ifdef NEWORDER_MPI
  no::Environment& env = no::getenv();
  // If rank=process, return the array, otherwise return an empty array
  if (source_dest.size() < (size_t)env.size())
    throw std::runtime_error("allgather array size %% is smaller than MPI size (%%)"_s % source_dest.size() % env.size());
  // take a copy of the soruce to avoid runtime error due to aliased buffers
  double source = no::at<double>(source_dest, {(size_t)env.rank()});
  double* p = no::begin<double>(source_dest);
  MPI_Allgather(&source, 1, mpi_type_trait<double>::type, p, 1, mpi_type_trait<double>::type, MPI_COMM_WORLD);
  return source_dest;
#else
  throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

void no::mpi::sync()
{
#ifdef NEWORDER_MPI
  //no::log("waiting for other processes...");
  MPI_Barrier(MPI_COMM_WORLD);
  //no::log("...resuming");
// #else
//   throw std::runtime_error("%% not implemented (binary doesn't support MPI)"_s % __FUNCTION__);
#endif
}

