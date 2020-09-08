
const char* version_docstr = R"docstr(
    Gets the module version
)docstr";

const char* log_docstr = R"docstr(
    The logging function. Prints obj to the console, annotated with process information
)docstr";

const char* verbose_docstr = R"docstr(
    Sets the verbose flag, which toggles detailed runtime logs
)docstr";

const char* checked_docstr = R"docstr(
    Sets the checked flag, which determines whether the model runs checks during execution
)docstr";

const char* run_docstr = R"docstr(
    Runs the model
    Returns:
        bool: True if model succeeded, False otherwise
)docstr";

const char* mpi_rank_docstr = R"docstr(
    The MPI rank of the process
    Returns:
        int: the MPI rank
)docstr";

const char* mpi_size_docstr = R"docstr(
    The MPI size (no. of processes) of the run
    Returns:
        int: the MPI size 
)docstr";

const char* time_distant_past_docstr = R"docstr(
    Returns a value that compares less than any other value but itself and "never"
    Returns:
        float: -inf
)docstr";

const char* time_far_future_docstr = R"docstr(
    Returns a value that compares greater than any other value but itself and "never"
    Returns:
        float: +inf
)docstr";

const char* time_never_docstr = R"docstr(
    Returns a value that compares unequal to any value, including but itself.
    Returns:
        float: nan
)docstr";

const char* time_isnever_docstr = R"docstr(
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN, 
    direct comparison will always fail, since NaN != NaN. 
    Args:
        t (float): The time.
    Returns:
        bool: True if t is never, False otherwise
)docstr";

const char* time_isnever_a_docstr = R"docstr(
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is 
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN. 
    Args:
        a (array(float)): The times.
    Returns:
        array(bool): True if corresponding input is never, False otherwise
)docstr";
