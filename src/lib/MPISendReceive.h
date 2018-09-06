

#pragma once 

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
struct mpi_type_trait<unsigned char>
{
  static const int type = MPI_UNSIGNED_CHAR;
};


//...

template<typename T>
void send(const T& data, int process)
{
  MPI_Send(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
}

template<typename T>
void send(const T& data, int len, int process)
{
  MPI_Send(&data, len, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
}

template<typename T>
void receive(T& data, int process)
{
  MPI_Recv(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template<typename T>
void receive(T& data, int len, int process)
{
  MPI_Recv(&data, len, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

#else

template<typename T>
void send(const T&, int)
{
  throw std::runtime_error("MPI not enabled");
}


template<typename T>
void send(const T&, int, int)
{
  throw std::runtime_error("MPI not enabled");
}

template<typename T>
void receive(T&, int)
{
  throw std::runtime_error("MPI not enabled");
}

template<typename T>
void receive(T&, int, int)
{
  throw std::runtime_error("MPI not enabled");
}


#endif

}} // neworder::mpi
