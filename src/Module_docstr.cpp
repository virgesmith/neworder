
const char* module_docstr = R"docstr(
A dynamic microsimulation framework";
)docstr";

// top-level functions

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
    Runs the model. If the model has previously run it will resume from the point at which it was given the "halt" instruction. This is useful
    for external processing of model data, and/or feedback from external sources. If the model has already reached the end of the timeline, this
    function will have no effect. To re-run the model from the start, you must construct a new model object.
    Returns:
        True if model succeeded, False otherwise
)docstr";

// Timeline

const char* lineartimeline_docstr = R"docstr(
    An equally-spaced non-calendar timeline .
)docstr";

const char* lineartimeline_init_docstr = R"docstr(
    Constructs a timeline from start to end, with the given number of steps.
)docstr";

const char* lineartimeline_init_open_docstr = R"docstr(
    Constructs an open-ended timeline give a start value and a step size. NB the model will run until the Model.halt() method is explicitly called
    (from inside the step() method). Note also that nsteps() will return -1 for timelines constructed this way
)docstr";

const char* numerictimeline_docstr = R"docstr(
    An custom non-calendar timeline where the user explicitly specifies the time points, which must be monotonically increasing.
)docstr";

const char* numerictimeline_init_docstr = R"docstr(
    Constructs a timeline from an array of time points.
)docstr";

const char* notimeline_docstr = R"docstr(
    An arbitrary one step timeline, for continuous-time models with no explicit (discrete) timeline
)docstr";

const char* notimeline_init_docstr = R"docstr(
    Constructs an arbitrary one step timeline, where the start and end times are undefined and there is a single step of size zero. Useful for continuous-time models
)docstr";

const char* calendartimeline_docstr = R"docstr(
    A calendar-based timeline
)docstr";

const char* calendartimeline_init_docstr = R"docstr(
    Constructs a calendar-based timeline, given start and end dates, an increment specified as a multiple of days, months or years
)docstr";

const char* calendartimeline_init_open_docstr = R"docstr(
    Constructs an open-ended calendar-based timeline, given a start date and an increment specified as a multiple of days, months or years.
     NB the model will run until the Model.halt() method is explicitly called (from inside the step() method). Note also that nsteps() will
     return -1 for timelines constructed this way
)docstr";

const char* timeline_start_docstr = R"docstr(
    Returns the time of the start of the timeline
)docstr";

const char* timeline_end_docstr = R"docstr(
    Returns the time of the end of the timeline
)docstr";

const char* timeline_index_docstr = R"docstr(
    Returns the index of the current step in the timeline
)docstr";

const char* timeline_time_docstr = R"docstr(
    Returns the time of the current step in the timeline
)docstr";

const char* timeline_dt_docstr = R"docstr(
    Returns the step size size of the timeline
)docstr";

const char* timeline_nsteps_docstr = R"docstr(
    Returns the number of steps in the timeline (or -1 if open-ended)
)docstr";

const char* timeline_at_end_docstr = R"docstr(
    Returns True if the current step is the end of the timeline
)docstr";

const char* timeline_repr_docstr = R"docstr(
    Prints a human-readable representation of the timeline
)docstr";

// MonteCarlo

const char* mc_docstr = R"docstr(
    The model's Monte-Carlo engine with configurable options for parallel execution
)docstr";

const char* mc_deterministic_identical_stream_docstr = R"docstr(
    Returns a deterministic seed (19937). Input argument is ignored
)docstr";

const char* mc_deterministic_independent_stream_docstr = R"docstr(
    Returns a deterministic seed that is a function of the input (19937+r).
    The model uses the MPI rank as the input argument, allowing for differently seeded streams in each process
)docstr";

const char* mc_nondeterministic_stream_docstr = R"docstr(
    Returns a random seed from the platform's random_device. Input argument is ignored
)docstr";

const char* mc_seed_docstr = R"docstr(
    Returns the seed used to initialise the random stream
)docstr";

const char* mc_reset_docstr = R"docstr(
    Resets the generator using the original seed.
    Use with care, esp in multi-process models with identical streams
)docstr";

const char* mc_state_docstr = R"docstr(
    Returns a hash of the internal state of the generator. Avoids the extra complexity of tranmitting variable-length strings over MPI.
)docstr";

const char* mc_raw_docstr = R"docstr(
    Returns a random 64-bit unsigned integer. Useful for seeding other generators.
)docstr";

const char* mc_ustream_docstr = R"docstr(
    Returns an array of uniform random [0,1) variates of length n
)docstr";

const char* mc_sample_docstr = R"docstr(
    Returns an array of length n containing randomly sampled categorical values, weighted according to cat_weights
)docstr";

const char* mc_hazard_docstr = R"docstr(
    Returns an array of ones (with hazard rate lambda) or zeros of length n
)docstr";

const char* mc_hazard_a_docstr = R"docstr(
    Returns an array of ones (with hazard rate lambda[i]) or zeros for each element in p
)docstr";

const char* mc_stopping_docstr = R"docstr(
    Returns an array of stopping times (with hazard rate lambda) of length n
)docstr";

const char* mc_stopping_a_docstr = R"docstr(
    Returns an array of stopping times (with hazard rate lambda[i]) for each element in lambda
)docstr";

const char* mc_counts_docstr = R"docstr(
    Returns an array of simulated arrival counts (within time dt) for each intensity in lambda
)docstr";

const char* mc_arrivals_docstr = R"docstr(
    Returns an array of n arrays of multiple arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with a minimum separation between events of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm
    The final value of lambda must be zero, and thus arrivals don't always occur, indicated by a value of neworder.time.never()
    The inner dimension of the returned 2d array is governed by the the maximum number of arrivals sampled, and will thus vary
)docstr";

const char* mc_first_arrival_docstr = R"docstr(
    Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with a minimum start time of minval. Sampling uses the Lewis-Shedler "thinning" algorithm
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)docstr";

const char* mc_first_arrival3_docstr = R"docstr(
    Returns an array of length n of first arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with no minimum start time. Sampling uses the Lewis-Shedler "thinning" algorithm
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)docstr";

const char* mc_next_arrival_docstr = R"docstr(
    Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with start times given by startingpoints with a minimum offset of mingap. Sampling uses the Lewis-Shedler "thinning" algorithm.
    If the relative flag is True, then lambda[0] corresponds to start time + mingap, not to absolute time
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)docstr";

const char* mc_next_arrival4_docstr = R"docstr(
    Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
    If the relative flag is True, then lambda[0] corresponds to start time, not to absolute time
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)docstr";

const char* mc_next_arrival3_docstr = R"docstr(
    Returns an array of length n of subsequent arrival times from a nonhomogeneous Poisson process (with hazard rate lambda[i], time interval dt),
    with start times given by startingpoints. Sampling uses the Lewis-Shedler "thinning" algorithm.
    If the final value of lambda is zero, no arrival is indicated by a value of neworder.time.never()
)docstr";

const char* mc_repr_docstr = R"docstr(
    Prints a human-readable representation of the MonteCarlo engine
)docstr";

// The Model class

const char* model_docstr = R"docstr(
    The base model class from which all neworder models should be subclassed
)docstr";

const char* model_init_notimeline_docstr = R"docstr(
    Constructs a model object with an empty timeline and a seeder function, for continuous-time models
)docstr";

const char* model_init_lineartimeline_docstr = R"docstr(
    Constructs a model object from a linear timeline and a seeder function, providing equally spaced timesteps
)docstr";

const char* model_init_numerictimeline_docstr = R"docstr(
    Constructs a model object from a numeric timeline and a seeder function, allowing user defined timesteps
)docstr";

const char* model_init_calendartimelime_docstr = R"docstr(
    Constructs a model object from a calendar timeline and a seeder function, with date-based timesteps
)docstr";

const char* model_timeline_docstr = R"docstr(
    Returns the model's timeline object
)docstr";
const char* model_mc_docstr = R"docstr(
    Returns the models Monte-Carlo engine
)docstr";
const char* model_modify_docstr = R"docstr(
    User-overridable method used to modify state in a per-process basis for multiprocess model runs.
    Default behaviour is to do nothing.
    This function should not be called directly, it is used by the Model.run() function
)docstr";
const char* model_step_docstr = R"docstr(
    User-implemented method used to advance state of a model.
    Default behaviour raises NotImplementedError.
    This function should not be called directly, it is used by the Model.run() function
)docstr";
const char* model_check_docstr = R"docstr(
    User-overridable method used to check internal state at each timestep.
    Default behaviour is to simply return True.
    Returning False will halt the model run.
    This function should not be called directly, it is used by the Model.run() function

    Returns:
        True if checks are ok, False otherwise.
)docstr";
const char* model_finalise_docstr = R"docstr(
    User-overridable function for custom processing after the final step in the model run.
    Default behaviour does nothing. This function does not need to be called directly, it is called by the Model.run() function
)docstr";
const char* model_halt_docstr = R"docstr(
    Signal to the model to stop execution gracefully at the end of the current timestep, e.g. if some convergence criterion has been met,
    or input is required from an upstream model. The model can be subsequently resumed by calling the run() function.
    For trapping exceptional/error conditions, prefer to raise an exception, or return False from the Model.check() function
)docstr";

// MPI

const char* mpi_docstr = R"docstr(
    Submodule for basic MPI environment discovery
)docstr";

const char* mpi_rank_docstr = R"docstr(
    Returns the MPI rank of the process
)docstr";

const char* mpi_size_docstr = R"docstr(
    Returns the MPI size (no. of processes) of the run
)docstr";

// Time

const char* time_docstr = R"docstr(
    Temporal values and comparison
)docstr";

const char* time_distant_past_docstr = R"docstr(
    Returns a value that compares less than any other value but itself and "never"
)docstr";

const char* time_far_future_docstr = R"docstr(
    Returns a value that compares greater than any other value but itself and "never"
)docstr";

const char* time_never_docstr = R"docstr(
    Returns a value that compares unequal to any value, including but itself.
)docstr";

const char* time_isnever_docstr = R"docstr(
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
    direct comparison will always fail, since NaN != NaN.
)docstr";

const char* time_isnever_a_docstr = R"docstr(
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.
)docstr";

// Statistical functions

const char* stats_docstr = R"docstr(
    Submodule for statistical functions
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

// Dataframe manipulation

const char* df_docstr = R"docstr(
    Submodule for operations involving direct manipulation of pandas dataframes
)docstr";


const char* df_unique_index_docstr = R"docstr(
    Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.
)docstr";


const char* df_transition_docstr = R"docstr(
    Randomly changes categorical data in a dataframe, according to supplied transition probabilities.
    Args:
        model: The model (for access to the MonteCarlo engine).
        categories: The set of possible categories
        transition_matrix: The probabilities of transitions between categories
        df: The dataframe, which is modified in-place
        colname: The name of the column in the dataframe
)docstr";

const char* df_testfunc_docstr = R"docstr(
    Test function for direct dataframe manipulation. Results may vary. Do not use.
)docstr";

