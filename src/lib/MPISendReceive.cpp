#include "MPISendReceive.h"

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
void send(const std::string& data, int process)
{
  int size = data.size();
  MPI_Send(&size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send length %%"_s % size);
  MPI_Send(data.data(), data.size(), mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send %%"_s % data.substr(40));

}


template<>
void send(const Buffer& data, int process)
{
  MPI_Send(&data.size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
  neworder::log("buf send length %%"_s % data.size);
  MPI_Send(data.buf, data.size, mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send %%"_s % data.substr(40));

}

template<>
void receive(std::string& data, int process)
{
  int size;
  MPI_Recv(&size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//  neworder::log("recv length %%"_s % size);

  data.resize(size);
  MPI_Recv(&data[0], size, mpi_type_trait<std::string>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void receive(Buffer& data, int process)
{
  int n;
  MPI_Recv(&n, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  neworder::log("buf recv length %%"_s % n);

  data.alloc(n);
  MPI_Recv(data.buf, data.size, mpi_type_trait<Buffer>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

#endif

}} // neworder::mpi


