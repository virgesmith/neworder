

#pragma once 

#include "Log.h"

namespace neworder {
namespace mpi {

void send(const py::object& o, int rank);

py::object receive(int rank);

void send_csv(const py::object& o, int rank);

py::object receive_csv(int rank);

// TODO make below private

// Buffer for MPI send/recv
struct Buffer
{
  typedef char value_type;

  // borrowed
  Buffer(char* b, int n);
  // owned
  Buffer(int n);

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer() ;

  void free();

  void alloc(int n);

  bool owned;
  char* buf;
  int size;
};

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
struct mpi_type_trait<Buffer>
{
  static const int type = MPI_CHAR;
};

template<>
struct mpi_type_trait<std::string>
{
  static const int type = MPI_CHAR;
};

template<>
struct mpi_type_trait<unsigned char>
{
  static const int type = MPI_UNSIGNED_CHAR;
};


template<typename T>
void send_impl(const T& data, int process)
{
  MPI_Send(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
}


template<typename T>
void receive_impl(T& data, int process)
{
  MPI_Recv(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<>
void send_impl(const std::string& data, int process);

template<>
void send_impl(const Buffer& data, int process);

template<>
void receive_impl(std::string& data, int process);

template<>
void receive_impl(Buffer& data, int process);

#else

template<typename T>
void send_impl(const T&, int)
{
  throw std::runtime_error("MPI not enabled");
}

template<typename T>
void receive_impl(T&, int)
{
  throw std::runtime_error("MPI not enabled");
}

#endif

}} // neworder::mpi
