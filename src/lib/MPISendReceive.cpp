#include "MPISendReceive.h"

#ifdef NEWORDER_MPI

#include <mpi.h>

namespace {

// Buffer for MPI send/recv
struct Buffer
{
  typedef char value_type;

  // borrowed
  Buffer(char* b, int n) : owned(false), buf(b), size(n) { }
  // owned
  Buffer(int n): owned(true), buf(new char[n]), size(n) { }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer()
  {
    free();
  }

  void alloc(int n)
  {
    free();
    owned = true;
    buf = new char[n];
    size = n;
  }

  void free()
  {
    if (owned)
      delete[] buf;
  }

  bool owned;
  char* buf;
  int size;
};

}

#ifdef NEWORDER_MPI
// these specialisation must be in global namespace (as per template)
template<>
struct mpi_type_trait<Buffer>
{
  static const int type = MPI_CHAR;
};

template<>
struct mpi_type_trait<std::string>
{
  static const int type = MPI_CHAR;
};
#endif

namespace {

// template<typename T>
// void send_impl(const T& data, int process)
// {
//   MPI_Send(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
// }


//template<>
void send_impl(const std::string& data, int process)
{
  // neworder::log("send length %%"_s % size);
  MPI_Send(data.data(), data.size(), mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD);
  // neworder::log("send %%"_s % data.substr(40));
}


//template<>
void send_impl(const Buffer& data, int process)
{
  //MPI_Send(&data.size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
  //neworder::log("buf send length %%"_s % data.size);
  MPI_Send(data.buf, data.size, mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD);
  // neworder::log("send %%"_s % data.substr(40));
}

// template<typename T>
// void receive_impl(T& data, int process)
// {
//   MPI_Status status;
//   MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

//   // When probe returns, the status object has the size and other attributes of the incoming message. Get the message size
//   int size;
//   MPI_Get_count(&status, mpi_type_trait<std::string>::type, &size);
//   MPI_Recv(&data, size, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// }

//template<>
void receive_impl(std::string& data, int process)
{
  // Probe for an incoming message from process zero
  MPI_Status status;
  MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

  // When probe returns, the status object has the size and other attributes of the incoming message. Get the message size
  int size;
  MPI_Get_count(&status, mpi_type_trait<std::string>::type, &size);
  neworder::log("str recv length %%"_s % size);

  data.resize(size);
  MPI_Recv(&data[0], size, mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

//template<>
void receive_impl(Buffer& data, int process)
{
  // Probe for an incoming message from process zero
  MPI_Status status;
  MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

  // When probe returns, the status object has the size and other attributes of the incoming message. Get the message size
  int size;
  MPI_Get_count(&status, mpi_type_trait<Buffer>::type, &size);
  //neworder::log("buf recv length %%"_s % size);

  data.alloc(size);
  MPI_Recv(data.buf, data.size, mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

} // anon

#endif


// to the next rank 
void neworder::mpi::send_obj(const py::object& o, int rank)
{
#ifdef NEWORDER_MPI
  py::object pickle = py::import("pickle");
  py::object serialised = pickle.attr("dumps")(o);

  //you can also write checks here for length, verify the 
  //buffer is memory-contiguous, etc.
  Buffer b(PyBytes_AsString(serialised.ptr()), (int)PyBytes_Size(serialised.ptr()));

  // something along the lines of (see https://docs.python.org/3.6/c-api/buffer.html)
  //std::string s(PyBytes_AsString(serialised.ptr()), py::len(serialised)); 
  //neworder::log("sending %% (len %%) to 1"_s % s % s.size());
  send_impl(b, rank);
#else
  throw std::runtime_error("cannot send: MPI not enabled");
#endif
}

py::object neworder::mpi::receive_obj(int rank)
{
#ifdef NEWORDER_MPI
  Buffer b(nullptr, 0);
  receive_impl(b, rank); // b is alloc'd in here

  py::object pickle = py::import("pickle");

  py::object o = pickle.attr("loads")(py::handle<>(PyBytes_FromStringAndSize(b.buf, b.size)));
  //neworder::log("got %% from %%"_s % s % rank);
  return o;
#else
  throw std::runtime_error("cannot recv: MPI not enabled");
#endif
}

void neworder::mpi::send_csv(const py::object& df, int rank)
{
#ifdef NEWORDER_MPI
  py::object io = py::import("io");
  py::object buf = io.attr("StringIO")();
  // kwargs broken?
  // py::dict kwargs;
  // kwargs["index"] = false;
  // to_csv(path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True...
  df.attr("to_csv")(buf); //, ", ", "", py::object(), py::object(), true, false); 
  std::string csvbuf = py::extract<std::string>(buf.attr("getvalue")())();
  send_impl(csvbuf, rank);
#else
  throw std::runtime_error("cannot recv: MPI not enabled");
#endif
}

py::object neworder::mpi::receive_csv(int rank)
{
#ifdef NEWORDER_MPI
  std::string buf;
  receive_impl(buf, rank);

  py::object io = py::import("io");
  py::object pd = py::import("pandas");
  py::object pybuf = io.attr("StringIO")(buf);
  py::object df = pd.attr("read_csv")(pybuf);
  // temp workaround for not being able to pass to_csv index=False arg
  df = df.attr("drop")("Unnamed: 0", 1);
  return df;
#else
  throw std::runtime_error("cannot recv: MPI not enabled");
#endif
}

void neworder::mpi::sync()
{
#ifdef NEWORDER_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}


//#include "Environment.h"

// Broadcast object from rank to all other procs
// Have to return by value as (some) python objects are immutable 
py::object neworder::mpi::broadcast_obj(py::object& o, int rank)
{
#ifdef NEWORDER_MPI
  int n = py::extract<int>(o)();
//  neworder::log("broadcast (%%) %%"_s % pycpp::Environment::get().rank() % n);
  broadcast(n, rank);
  return py::object(n);
#else
  return o;
#endif
}

