
const char* version_docstr = R"docstr(
    The version function

    Args:

    Returns:
    str: the module version
)docstr";

const char* log_docstr = R"pydocstr(
    The logging function. Prints x to the console, annotated with process information

    Args:
    x: object

    Returns:
    None
)pydocstr";

const char* mpi_rank_docstr = R"pydocstr(
    Returns the MPI rank of the process

    Args:

    Returns:
    int: the MPI rank
)pydocstr";

const char* mpi_size_docstr = R"pydocstr(
    Returns the MPI size (no. of processes) of the run

    Args:

    Returns:
    int: the MPI size 
)pydocstr";

