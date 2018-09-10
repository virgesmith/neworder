

#pragma once 

#include "Log.h"

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

#endif

namespace neworder {
namespace mpi {

void send_obj(const py::object& o, int rank);

py::object receive_obj(int rank);

void send_csv(const py::object& o, int rank);

py::object receive_csv(int rank);

template<typename T>
void send(const T& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Send(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
#else
#endif
}

template<typename T>
void receive(T& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Recv(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
#endif
}

// template<>
// void send_impl(const std::string& data, int process);

// template<>
// void send_impl(const Buffer& data, int process);

// template<>
// void receive_impl(std::string& data, int process);

// template<>
// void receive_impl(Buffer& data, int process);

// #else

// // template<typename T>
// // void send_impl(const T&, int)
// // {
// //   throw std::runtime_error("MPI not enabled");
// // }

// // template<typename T>
// // void receive_impl(T&, int)
// // {
// //   throw std::runtime_error("MPI not enabled");
// // }

// #endif

}} // neworder::mpi
