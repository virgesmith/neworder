

#pragma once 

#include "Log.h"
#include "Environment.h"

#ifdef NEWORDER_MPI
#include <mpi.h>

// The type of e.g. MPI_INT differs between implementations
template<typename T> 
struct mpi_type_trait;

template<>
struct mpi_type_trait<bool>
{
  static constexpr const auto type = MPI_INT;
};

template<>
struct mpi_type_trait<int>
{
  static constexpr const auto type = MPI_INT;
};

template<>
struct mpi_type_trait<int64_t>
{
  static constexpr const auto type = MPI_LONG_LONG_INT;
};

template<>
struct mpi_type_trait<char>
{
  static constexpr const auto type = MPI_CHAR;
};

template<>
struct mpi_type_trait<unsigned char>
{
  static constexpr const auto type = MPI_UNSIGNED_CHAR;
};

template<>
struct mpi_type_trait<double>
{
  static constexpr const auto type = MPI_DOUBLE;
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

// Gather scalars from each process into a numpy array on process rank
np::ndarray gather_array(double x, int rank);

// Gather scalars from each process into a numpy array on process rank
double scatter_array(np::ndarray x, int rank);

// TODO more efficient with references?
np::ndarray allgather_array(np::ndarray source_dest);

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


template<typename T>
std::vector<T>& gather(const T& source, std::vector<T>& dest, int process)
{
#ifdef NEWORDER_MPI
  neworder::Environment& env = neworder::getenv();
  // If rank=process, return the array, otherwise return an empty array
  T* p = nullptr;
  if (env.rank() == process)
  {
    dest.resize(env.size());
    p = dest.data();
  }
  else
  {
    dest.clear();    
  }
  MPI_Gather(&source, 1, mpi_type_trait<T>::type, p, 1, mpi_type_trait<T>::type, process, MPI_COMM_WORLD);
#endif
  return dest;
}

template<typename T>
T& scatter(const std::vector<T>& source, T& dest, int process)
{
#ifdef NEWORDER_MPI
  neworder::Environment& env = neworder::getenv();
  // If rank=process, return the array, otherwise return an empty array
  T* p = nullptr;
  if (env.rank() == process)
  {
    if (source.size() < (size_t)env.size())
      throw std::runtime_error("scatter array size %% is smaller than MPI size (%%)"_s % source.size() % env.size());
    p = const_cast<T*>(source.data());
  }
  MPI_Scatter(p, 1, mpi_type_trait<T>::type, &dest, 1, mpi_type_trait<T>::type, process, MPI_COMM_WORLD);
#endif
  return dest;
}

template<typename T>
std::vector<T>& allgather(std::vector<T>& source_dest)
{
#ifdef NEWORDER_MPI
  neworder::Environment& env = neworder::getenv();
  // If rank=process, return the array, otherwise return an empty array
  if (source_dest.size() < (size_t)env.size())
    throw std::runtime_error("allgather array size %% is smaller than MPI size (%%)"_s % source_dest.size() % env.size());
  // take a copy of the soruce to avoid runtime error due to aliased buffers
  T source = source_dest[env.rank()];
  T* p = source_dest.data();
  MPI_Allgather(&source, 1, mpi_type_trait<T>::type, p, 1, mpi_type_trait<T>::type, MPI_COMM_WORLD);
#endif
  return source_dest;
}


void sync();

}} // neworder::mpi
