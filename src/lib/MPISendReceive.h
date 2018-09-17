

#pragma once 

#include "Log.h"

#ifdef NEWORDER_MPI
#include <mpi.h>

template<typename T> 
struct mpi_type_trait;

template<>
struct mpi_type_trait<bool>
{
  static const int type = MPI_INT;
};

template<>
struct mpi_type_trait<int>
{
  static const int type = MPI_INT;
};

template<>
struct mpi_type_trait<int64_t>
{
  static const int type = MPI_LONG_LONG_INT;
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

template<>
struct mpi_type_trait<double>
{
  static const int type = MPI_DOUBLE;
};

#endif

namespace neworder {
namespace mpi {

void send_obj(const py::object& o, int rank);

py::object receive_obj(int rank);

void send_csv(const py::object& o, int rank);

py::object receive_csv(int rank);

// Broadcast object from rank to all other procs
py::object broadcast_obj(py::object& o, int rank);

template<typename T>
void send(const T& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Send(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD);
#endif
}

template<>
void send(const std::string& data, int process);

template<typename T>
void receive(T& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Recv(&data, 1, mpi_type_trait<T>::type, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
}

template<>
void receive(std::string& data, int process);

template<typename T>
void broadcast(T& data, int process)
{
#ifdef NEWORDER_MPI
  MPI_Bcast(&data, 1, mpi_type_trait<T>::type, process, MPI_COMM_WORLD);
#endif
}

template<>
void broadcast(std::string& data, int process);

void sync();

}} // neworder::mpi
