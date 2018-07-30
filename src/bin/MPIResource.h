
#pragma once 

#include <mpi.h>
//#include <mpicxx.h> ???

#include <string>
#include <stdexcept>

using namespace MPI;

// TODO move definitions to cpp (will require reworking Makefile)

// RAII wrapper for MPI initialisation
class MPIResource
{
public:

	MPIResource(int* pargc, char*** pargv)
	{
		// TODO check return values?
		int status = MPI_Init(pargc, pargv);
		if (status != MPI_SUCCESS)
		{
		  throw std::runtime_error("MPI init failed, error: " + std::to_string(status));
		}
		
		status = MPI_Comm_size(MPI_COMM_WORLD, &m_worldSize);
		if (status != MPI_SUCCESS)
		{
		  throw std::runtime_error("MPI size failed, error: " + std::to_string(status));
		}

  	// Get the rank of the process
  	status = MPI_Comm_rank(MPI_COMM_WORLD, &m_worldRank);
		if (status != MPI_SUCCESS)
		{
		  throw std::runtime_error("MPI rank failed, error: " + std::to_string(status));
		}

  	// Get the name of the processor
  	int name_len;
	  status = MPI_Get_processor_name(m_processorName, &name_len);		
		if (status != MPI_SUCCESS)
		{
		  throw std::runtime_error("MPI name failed, error: " + std::to_string(status));
		}
	}

	~MPIResource()
	{
		// Finalize the MPI environment. Return value always MPI_SUCCESS (which is good as we can't throw)
		MPI_Finalize();	
	}
	
	// Disable copy/assign
	MPIResource(const MPIResource&) = delete;
	MPIResource& operator=(const MPIResource&) = delete;

	int rank() const { return m_worldRank; }
	
	int size() const { return m_worldSize; }
	
	const char* name() const { return m_processorName; }
	
private:
	int m_worldRank;
	int m_worldSize;
	char m_processorName[MPI_MAX_PROCESSOR_NAME];
};

