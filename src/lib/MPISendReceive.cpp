#include "MPISendReceive.h"


// to the next rank 
void neworder::mpi::send(const py::object& o, int rank)
{
  py::object pickle = py::import("pickle");
  py::object serialised = pickle.attr("dumps")(o);

  //you can also write checks here for length, verify the 
  //buffer is memory-contiguous, etc.
  mpi::Buffer b(PyBytes_AsString(serialised.ptr()), (int)PyBytes_Size(serialised.ptr()));

  // something along the lines of (see https://docs.python.org/3.6/c-api/buffer.html)
  //std::string s(PyBytes_AsString(serialised.ptr()), py::len(serialised)); 
  //neworder::log("sending %% (len %%) to 1"_s % s % s.size());
  neworder::mpi::send_impl(b, rank);
}

py::object neworder::mpi::receive(int rank)
{
  mpi::Buffer b(nullptr, 0);
  neworder::mpi::receive_impl(b, rank); // b is alloc'd in here

  py::object pickle = py::import("pickle");

  py::object o = pickle.attr("loads")(py::handle<>(PyBytes_FromStringAndSize(b.buf, b.size)));
  //neworder::log("got %% from %%"_s % s % rank);
  return o;
}

void neworder::mpi::send_csv(const py::object& df, int rank)
{
  py::object io = py::import("io");
  py::object buf = io.attr("StringIO")();
  // kwargs broken?
  // py::dict kwargs;
  // kwargs["index"] = false;
  // to_csv(path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True...
  df.attr("to_csv")(buf); //, ", ", "", py::object(), py::object(), true, false); 
  std::string csvbuf = py::extract<std::string>(buf.attr("getvalue")())();
  neworder::mpi::send_impl(csvbuf, rank);
}

py::object neworder::mpi::receive_csv(int rank)
{
  std::string buf;
  neworder::mpi::receive_impl(buf, rank);

  py::object io = py::import("io");
  py::object pd = py::import("pandas");
  py::object pybuf = io.attr("StringIO")(buf);
  py::object df = pd.attr("read_csv")(pybuf);
  // temp workaround for not being able to pass to_csv index=False arg
  df = df.attr("drop")("Unnamed: 0", 1);
  return df;
}


// gcc bug (<7) means we have to declare specialisations within namespace rather than qualifying the specialisation
namespace neworder { namespace mpi {

Buffer::Buffer(char* b, int n) : owned(false), buf(b), size(n) { } 

Buffer::Buffer(int n) : owned(true), buf(new char[n]), size(n) { }

Buffer::~Buffer() 
{
  free();
}

void Buffer::free()
{
  if (owned)
    delete[] buf;
}

void Buffer::alloc(int n)
{
  free();
  owned = true;
  buf = new char[n];
  size = n;
}

#ifdef NEWORDER_MPI

template<>
void send_impl(const std::string& data, int process)
{
  int size = data.size();
  MPI_Send(&size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send length %%"_s % size);
  MPI_Send(data.data(), data.size(), mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send %%"_s % data.substr(40));

}


template<>
void send_impl(const Buffer& data, int process)
{
  MPI_Send(&data.size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
  neworder::log("buf send length %%"_s % data.size);
  MPI_Send(data.buf, data.size, mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send %%"_s % data.substr(40));

}

template<>
void receive_impl(std::string& data, int process)
{
  int size;
  MPI_Recv(&size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//  neworder::log("recv length %%"_s % size);

  data.resize(size);
  MPI_Recv(&data[0], size, mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void receive_impl(Buffer& data, int process)
{
  int n;
  MPI_Recv(&n, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  neworder::log("buf recv length %%"_s % n);

  data.alloc(n);
  MPI_Recv(data.buf, data.size, mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

#endif

}} // neworder::mpi


