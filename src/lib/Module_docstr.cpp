
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
        True if model succeeded, False otherwise
)docstr";

const char* mpi_rank_docstr = R"docstr(
    Returns the MPI rank of the process
)docstr";

const char* mpi_size_docstr = R"docstr(
    Returns the MPI size (no. of processes) of the run
)docstr";

const char* time_distant_past_docstr = R"docstr(
    Returns a value that compares less than any other value but itself and "never"
    Returns:
        -inf
)docstr";

const char* time_far_future_docstr = R"docstr(
    Returns a value that compares greater than any other value but itself and "never"
    Returns:
        +inf
)docstr";

const char* time_never_docstr = R"docstr(
    Returns a value that compares unequal to any value, including but itself.
    Returns:
        nan
)docstr";

const char* time_isnever_docstr = R"docstr(
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN, 
    direct comparison will always fail, since NaN != NaN. 
    Args:
        t: The time.
    Returns:
        True if t is never, False otherwise
)docstr";

const char* time_isnever_a_docstr = R"docstr(
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is 
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN. 
    Args:
        t: The times.
    Returns:
        Booleans, True where corresponding input value is never, False otherwise
)docstr";


const char* stats_logistic_docstr = R"docstr(
    Computes the logistic function on the supplied values. 
    Args:
        x: The input values.
        k: The growth rate
        x0: the midpoint location
    Returns:
        The function values
)docstr";

const char* stats_logistic_docstr_2 = R"docstr(
    Computes the logistic function with x0=0 on the supplied values. 
    Args:
        x: The input values.
        k: The growth rate
    Returns:
        The function values
)docstr";

const char* stats_logistic_docstr_1 = R"docstr(
    Computes the logistic function with k=1 and x0=0 on the supplied values. 
    Args:
        x: The input values.
    Returns:
        The function values
)docstr";

const char* stats_logit_docstr = R"docstr(
    Computes the logit function on the supplied values. 
    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
)docstr";

// temporary
const char* empty_docstr = R"docstr(
)docstr";
