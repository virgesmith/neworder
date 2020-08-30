### Population Microsimulation (Parallel)

The above model has been modified to run in massively parallel mode using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), for the entire population of England & Wales (approx 56 million people as of 2011 census). The input data is not under source control due to its size, but the 348 input files (one per local authority) are divided roughly equally over the MPI processes. This particular example, with its simple in-out migration model, lends itself easily to parallel execution as no interprocess communication is required. Future development of this package will enable interprocess communication, for e.g. moving people from one region to another.

The microsimulation has been run on the ARC3[[2]](#references) cluster and took a little over 4 minutes on 48 cores to simulate the population over a 40 year period.

See the [examples/people_multi](examples/people_multi) directory and the script [mpi_job.sh](mpi_job.sh)

