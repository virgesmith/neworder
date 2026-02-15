
const char* module_docstr = R"""(
A dynamic microsimulation framework";
)""";

// top-level functions

const char* freethreaded_docstr = R"""(
    Returns whether neworder was *built* with free-threading support (i.e. no GIL).

    Note: Other packages (e.g. pandas) may re-enable the GIL at runtime. To check, use `sys._is_gil_enabled()`. To force
    free-threading (at your own risk), use PYTHON_GIL=0 or -Xgil=0.
)""";

const char* threadid_docstr = R"""(
    Returns a unique thread id - equivalent to threading.get_native_id(). Use for control flow with extreme caution -
    order of thread initialisation cannot be guaranteed, and the python runtime may reuse completed threads.
)""";

const char* log_docstr = R"""(
    The logging function. Prints *args to the console, annotated with process/thread information
)""";

const char* verbose_docstr = R"""(
    Sets the verbose flag, which toggles detailed runtime logs
)""";

const char* checked_docstr = R"""(
    Sets the checked flag, which determines whether the model runs checks during execution
)""";

const char* run_docstr = R"""(
    Runs the model. If the model has previously run it will resume from the point at which it was given the "halt" instruction. This is useful
    for external processing of model data, and/or feedback from external sources. If the model has already reached the end of the timeline, this
    function will have no effect. To re-run the model from the start, you must construct a new model object.
    Returns:
        True if model succeeded, False otherwise
)""";

// Timeline

const char* lineartimeline_docstr = R"""(
    An equally-spaced non-calendar timeline .
)""";

const char* lineartimeline_init_docstr = R"""(
    Constructs a timeline from start to end, with the given number of steps.
)""";

const char* lineartimeline_init_open_docstr = R"""(
    Constructs an open-ended timeline give a start value and a step size. NB the model will run until the Model.halt() method is explicitly called
    (from inside the step() method). Note also that nsteps() will return -1 for timelines constructed this way
)""";

const char* numerictimeline_docstr = R"""(
    An custom non-calendar timeline where the user explicitly specifies the time points, which must be monotonically increasing.
)""";

const char* numerictimeline_init_docstr = R"""(
    Constructs a timeline from an array of time points.
)""";

const char* notimeline_docstr = R"""(
    An arbitrary one step timeline, for continuous-time models with no explicit (discrete) timeline
)""";

const char* notimeline_init_docstr = R"""(
    Constructs an arbitrary one step timeline, where the start and end times are undefined and there is a single step of size zero. Useful for continuous-time models
)""";

const char* calendartimeline_docstr = R"""(
    A calendar-based timeline
)""";

const char* calendartimeline_init_docstr = R"""(
    Constructs a calendar-based timeline, given start and end dates, an increment specified as a multiple of days, months or years
)""";

const char* calendartimeline_init_open_docstr = R"""(
    Constructs an open-ended calendar-based timeline, given a start date and an increment specified as a multiple of days, months or years.
     NB the model will run until the Model.halt() method is explicitly called (from inside the step() method). Note also that nsteps() will
     return -1 for timelines constructed this way
)""";

const char* timeline_start_docstr = R"""(
    Returns the time of the start of the timeline
)""";

const char* timeline_end_docstr = R"""(
    Returns the time of the end of the timeline
)""";

const char* timeline_index_docstr = R"""(
    Returns the index of the current step in the timeline
)""";

const char* timeline_time_docstr = R"""(
    Returns the time of the current step in the timeline
)""";

const char* timeline_dt_docstr = R"""(
    Returns the step size size of the timeline
)""";

const char* timeline_nsteps_docstr = R"""(
    Returns the number of steps in the timeline (or -1 if open-ended)
)""";

const char* timeline_at_end_docstr = R"""(
    Returns True if the current step is the end of the timeline
)""";

const char* timeline_repr_docstr = R"""(
    Prints a human-readable representation of the timeline object
)""";

// MonteCarlo

const char* mc_docstr = R"""(
    The model's Monte-Carlo engine with configurable options for parallel execution
)""";

const char* mc_deterministic_identical_stream_docstr = R"""(
    Returns a deterministic seed (19937).
)""";

const char* mc_deterministic_independent_stream_docstr = R"""(
    Returns a deterministic seed that is a function of the process rank (19937+r).
    Each process will have independent streams. Threads within the process will have identical streams.
)""";

const char* mc_nondeterministic_stream_docstr = R"""(
    Returns a random seed from the platform's random_device.
)""";

const char* mc_seed_docstr = R"""(
    Returns the seed used to initialise the random stream
)""";

const char* mc_reset_docstr = R"""(
    Resets the generator using the original seed.
    Use with care, esp in multi-process models with identical streams
)""";

const char* mc_state_docstr = R"""(
    Returns a hash of the internal state of the generator. Avoids the extra complexity of tranmitting variable-length strings over MPI.
)""";

const char* mc_raw_docstr = R"""(
    Returns a random 64-bit unsigned integer. Useful for seeding other generators.
)""";

const char* mc_ustream_docstr = R"""(
    Returns an array of uniform random [0,1) variates of length n
)""";

const char* mc_sample_docstr = R"""(
    Returns an array of length n containing randomly sampled categorical values, weighted according to cat_weights
)""";

const char* mc_hazard_docstr = R"""(
    Returns an array of ones (with hazard rate lambda) or zeros of length n
)""";

const char* mc_hazard_a_docstr = R"""(
    Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p
)""";

const char* mc_stopping_docstr = R"""(
    Returns an array of stopping times (with hazard rate lambda) of length n
)""";

const char* mc_stopping_a_docstr = R"""(
    Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda
)""";

const char* mc_counts_docstr = R"""(
    Returns an array of simulated arrival counts (within time dt) for each intensity in lambda
)""";

const char* mc_arrivals_docstr = R"""(
    Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
    The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
    The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary
)""";

const char* mc_first_arrival_docstr = R"""(
    Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with an optional minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)""";

const char* mc_next_arrival_docstr = R"""(
    Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
    If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)""";

const char* mc_repr_docstr = R"""(
    Prints a human-readable representation of the MonteCarlo engine
)""";

// The Model class

const char* model_docstr = R"""(
    The base model class from which all neworder models should be subclassed
)""";

const char* model_init_docstr = R"""(
    Constructs a model object with a timeline and (optionally) a seeder function for the random stream(s)
)""";

const char* model_timeline_docstr = R"""(
    The model's timeline object
)""";
const char* model_mc_docstr = R"""(
    The model's Monte-Carlo engine
)""";
const char* model_runstate_docstr = R"""(
    The model's current state - one of:
        NOT_STARTED: model has not been run
        RUNNING: model is in progress
        HALTED: model has been explicitly halted by calling its halt() method
        COMPLETED: model has run to the end of its timeline
)""";
const char* model_modify_docstr = R"""(
    User-overridable method used to modify state in a per-process basis for multiprocess model runs.
    Default behaviour is to do nothing.
    This function should not be called directly, it is used by the Model.run() function
)""";
const char* model_step_docstr = R"""(
    User-implemented method used to advance state of a model.
    Default behaviour raises NotImplementedError.
    This function should not be called directly, it is used by the Model.run() function
)""";
const char* model_check_docstr = R"""(
    User-overridable method used to check internal state at each timestep.
    Default behaviour is to simply return True.
    Returning False will halt the model run.
    This function should not be called directly, it is used by the Model.run() function

    Returns:
        True if checks are ok, False otherwise.
)""";
const char* model_finalise_docstr = R"""(
    User-overridable function for custom processing after the final step in the model run.
    Default behaviour does nothing. This function does not need to be called directly, it is called by the Model.run() function
)""";
const char* model_halt_docstr = R"""(
    Signal to the model to stop execution gracefully at the end of the current timestep, e.g. if some convergence criterion has been met,
    or input is required from an upstream model. The model can be subsequently resumed by calling the run() function.
    For trapping exceptional/error conditions, prefer to raise an exception, or return False from the Model.check() function
)""";

// MPI

const char* mpi_docstr = R"""(
    Submodule for basic MPI environment discovery, containing the following attributes:

    RANK: the process rank (0 in serial mode)
    SIZE: the number of processes (1 in serial mode)
    COMM: the MPI communicator (None in serial mode)
)""";


// Time

const char* time_docstr = R"""(
    Temporal values and comparison, including the attributes:
    NEVER: a value that compares unequal to any value, including itself.
    DISTANT_PAST: a value that compares less than any other value but itself and NEVER
    FAR_FUTURE: a value that compares greater than any other value but itself and NEVER
)""";

const char* time_isnever_docstr = R"""(
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
    direct comparison will always fail, since NaN != NaN.
)""";

const char* time_isnever_a_docstr = R"""(
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.
)""";

// Statistical functions

const char* stats_docstr = R"""(
    Submodule for statistical functions
)""";

const char* stats_logistic_docstr = R"""(
    Computes the logistic function on the supplied values.
    Args:
        x: The input values.
        x0: the midpoint location (default 0)
        k: The growth rate (1/scale, default 1)
    Returns:
        The function values
)""";


const char* stats_logit_docstr = R"""(
    Computes the logit function on the supplied values.
    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
)""";

// Dataframe manipulation

const char* df_docstr = R"""(
    Submodule for operations involving direct manipulation of pandas dataframes
)""";


const char* df_unique_index_docstr = R"""(
    Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.
    When multiple threads are in use, specific index values should not be relied on as they are generally nondeterministic
)""";


const char* df_transition_docstr = R"""(
    Randomly changes categorical data in a dataframe, according to supplied transition probabilities.
    Args:
        model: The model (for access to the MonteCarlo engine).
        categories: The set of possible categories
        transition_matrix: The probabilities of transitions between categories
        df: The dataframe, which is modified in-place
        colname: The name of the column in the dataframe
)""";

const char* df_testfunc_docstr = R"""(
    Test function for direct dataframe manipulation. Results may vary. Do not use.
)""";

