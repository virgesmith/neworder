

#pragma once 

#include "Log.h"

namespace neworder {
namespace mpi {

#ifdef NEWORDER_MPI

#include <mpi.h>

template<typename T> 
struct mpi_type_trait;

template<>
struct mpi_type_trait<int>
{
  static const int type = MPI_INT;
};

template<>
struct mpi_type_trait<char>
{
  static const int type = MPI_CHAR;
};

template<>
struct mpi_type_trait<unsigned char>
{
  static const int type = MPI_UNSIGNED_CHAR;
};


//...
struct Buffer
{
  static const int MPITYPE = MPI_CHAR;
  Buffer() : owned(false), buf(nullptr), size(0) { }
  Buffer(char* b, int n) : owned(false), buf(b), size(n) { } 
  Buffer(int n) : owned(true), buf(new char[n]), size(n) { }
  ~Buffer() 
  {
    // TODO... confim python is deleting this? make it safe
    // if (owned)
    //   delete[] buf;
  }

  bool owned;
  char* buf;
  int size;
};

template<typename T>
void send(const T& data, int process)
{
  MPI_Send(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
}

template<>
void send(const std::string& data, int process)
{
  int size = data.size();
  MPI_Send(&size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send length %%"_s % size);
  MPI_Send(data.data(), data.size(), mpi_type_trait<std::string::value_type>::type, process, 0, MPI_COMM_WORLD);
//  neworder::log("send %%"_s % data.substr(40));

}


template<>
void send(const Buffer& data, int process)
{
  MPI_Send(&data.size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD);
  neworder::log("buf send length %%"_s % data.size);
  MPI_Send(data.buf, data.size, Buffer::MPITYPE, process, 0, MPI_COMM_WORLD);
//  neworder::log("send %%"_s % data.substr(40));

}

template<typename T>
void receive(T& data, int process)
{
  MPI_Recv(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void receive(std::string& data, int process)
{
  int size;
  MPI_Recv(&size, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//  neworder::log("recv length %%"_s % size);

  data.resize(size);
  MPI_Recv(&data[0], size, mpi_type_trait<std::string::value_type>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void receive(Buffer& data, int process)
{
  int n;
  MPI_Recv(&n, 1, mpi_type_trait<int>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  neworder::log("buf recv length %%"_s % n);

  data = Buffer(n);
  MPI_Recv(data.buf, data.size, Buffer::MPITYPE, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

#else

// Stub the functions for non-mpi builds
//...
struct Buffer
{
  static const int MPITYPE = -1;
  Buffer(int n = 0) : buf(nullptr), size(n) { }
  Buffer(char* b, int n) : buf(b), size(n) { }
  char* buf;
  int size;
};

template<typename T>
void send(const T&, int)
{
  throw std::runtime_error("MPI not enabled");
}

template<typename T>
void receive(T&, int)
{
  throw std::runtime_error("MPI not enabled");
}

#endif

}} // neworder::mpi
