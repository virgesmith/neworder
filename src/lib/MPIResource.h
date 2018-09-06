
#pragma once 

#ifdef NEWORDER_MPI

#include <mpi.h>
//#include <mpicxx.h> ???


// RAII wrapper for MPI initialisation
class MPIResource
{
public:

	MPIResource(int* pargc, const char*** pargv);

	~MPIResource();
	
	// Disable copy/assign
	MPIResource(const MPIResource&) = delete;
	MPIResource& operator=(const MPIResource&) = delete;

	int rank() const;
	
	int size() const;
	
	const char* name() const;
	
private:
	int m_worldRank;
	int m_worldSize;
	char m_processorName[MPI_MAX_PROCESSOR_NAME];
};

#endif