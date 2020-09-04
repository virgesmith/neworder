
const char* version_docstr = R"docstr(
    Gets the module version
    Args:
    None
    Returns:
    str: the module version
)docstr";

const char* log_docstr = R"pydocstr(
    The logging function. Prints obj to the console, annotated with process information
    Args:
    obj: object
    Returns:
    None
)pydocstr";

const char* verbose_docstr = R"pydocstr(
    Sets the verbose flag, which toggles detailed runtime logs
    Args:
    verbose: bool
    Returns:
    None
)pydocstr";

const char* checked_docstr = R"pydocstr(
    Sets the checked flag, which determines whether the model runs checks during execution
    Args:
    verbose: bool
    Returns:
    None
)pydocstr";

const char* run_docstr = R"pydocstr(
    Runs the model
    Args:
    model: Model
    Returns:
    bool: True if model succeeded, False otherwise
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

const char* time_isnever_docstr = R"pydocstr(
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN, 
    direct comparison will always fails, since NaN != NaN. 
    Args:
        t (float): The time.

    Returns:
        bool: True if t is never, False otherwise
)pydocstr";

const char* time_isnever_a_docstr = R"pydocstr(
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is 
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN. 
    Args:
        a (array(float)): The times.

    Returns:
        array(bool): True if corresponding input is never, False otherwise
)pydocstr";
