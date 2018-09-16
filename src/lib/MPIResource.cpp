
#ifdef NEWORDER_MPI

#include "MPIResource.h"
#include "Log.h"

//#include <string>
#include <stdexcept>

//using namespace MPI;

// RAII wrapper for MPI initialisation
MPIResource::MPIResource(int* pargc, const char*** pargv)
{
  int status = MPI_Init(pargc, const_cast<char***>(pargv));
  if (status != MPI_SUCCESS)
  {
    throw std::runtime_error("MPI init failed, error: %%"_s % status);
  }
  
  status = MPI_Comm_size(MPI_COMM_WORLD, &m_worldSize);
  if (status != MPI_SUCCESS)
  {
    throw std::runtime_error("MPI size failed, error: %%"_s % status);
  }

  // Get the rank of the process
  status = MPI_Comm_rank(MPI_COMM_WORLD, &m_worldRank);
  if (status != MPI_SUCCESS)
  {
    throw std::runtime_error("MPI rank failed, error: "_s % status);
  }

  // Get the name of the processor
  int name_len;
  status = MPI_Get_processor_name(m_processorName, &name_len);		
  if (status != MPI_SUCCESS)
  {
    throw std::runtime_error("MPI name failed, error: %%"_s % status);
  }
}

MPIResource::~MPIResource()
{
  // Finalize the MPI environment. Return value always MPI_SUCCESS (which is good as we can't throw)
  MPI_Finalize();	
}
	
int MPIResource::rank() const { return m_worldRank; }

int MPIResource::size() const { return m_worldSize; }

const char* MPIResource::name() const { return m_processorName; }
	
#endif
