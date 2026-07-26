
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
    (from inside the step() method).
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

const char* timeline_at_end_docstr = R"""(
    Returns True if the current step is the end of the timeline
)""";

const char* timeline_repr_docstr = R"""(
    Prints a human-readable representation of the timeline object
)""";

// SplitMix64

const char* sms_docstr = R"""(
    A hash-based sampler that produces U[0,1) variates deterministically from integer keys.

    Uses the SplitMix64 finalizer (Stafford Variant 13) to hash an arbitrary sequence of
    integer arguments into a float64 value. Unlike a stream-based PRNG, the output at any
    index depends only on the seed and the input keys, so draws are independent of call order.
    When use_counter=False (the default), instances are safe to share across threads.
    When use_counter=True, the internal counter is updated atomically, but concurrent callers
    will interleave counter values non-deterministically — use one instance per thread if
    reproducible counter sequencing is required.

    uarray() accepts any number of positional arguments, each either a scalar int or a 1-D
    integer array. All scalar args (regardless of position) are premixed into a shared context
    hash (the "salt") before any array elements are processed. Array args each contribute one
    dimension to the output (outer-product semantics), and are hashed on top of the salt.
)""";

const char* sms_init_docstr = R"""(
    Constructs a SplitMix64 with a seeder callable and an optional call counter.

    The seeder is called immediately to set the initial seed and again on each reset().
    When use_counter=True, a monotonically increasing counter is mixed into the hash before
    any user-supplied arguments, guaranteeing that successive uarray() calls with identical
    arguments produce independent draws.

    Args:
        seeder: A zero-argument callable returning an integer seed.
        use_counter: (keyword-only) If True, advance an internal counter on each uarray() call (default False).
)""";

const char* sms_counter_docstr = R"""(
    The current call counter (uint64). Incremented by each uarray() call when use_counter=True;
    always 0 otherwise. Reset to 0 by reset().
)""";

const char* sms_reset_docstr = R"""(
    Resets the call counter to zero. The seeder is called fresh on each uarray() call,
    so reset() only affects the counter.
)""";

const char* sms_hash64_docstr = R"""(
    Returns a deterministic 64-bit integer hash of a string.

    Uses FNV-1a to accumulate the string bytes, then applies the SplitMix64 finalizer
    to diffuse the bits. The result is stable across platforms and Python versions and
    can be passed directly as a scalar key to uarray().

    Args:
        s: The string to hash.

    Returns:
        A signed 64-bit integer.
)""";

const char* sms_uarray_docstr = R"""(
    Returns a float64 array of U[0,1) values hashed from the supplied integer keys.

    Each positional argument is either a scalar int or a 1-D integer array:
      - Scalar args (in argument order) are premixed into a shared salt before any array
        elements are processed. They do not add an output dimension.
      - Array args (in argument order) are each folded into the hash on top of the salt,
        each adding one output dimension (outer-product semantics).

    The value at any output index depends only on the seed, the call counter (if enabled),
    and the corresponding input key values - not on position within the array or which other
    keys are present. This makes draws safe to use under sub-sampling and reordering.

    Args:
        *args: One or more scalar ints or 1-D integer arrays.

    Returns:
        ndarray[float64] with shape (len(arr0), len(arr1), ...) for the array args in order.
        A 0-d array is returned when all args are scalars.

    Raises:
        ValueError: If no arguments are supplied.
        TypeError: If any argument is not a scalar int or a 1-D integer array.
)""";

const char* sms_repr_docstr = R"""(
    Returns a human-readable representation of the SplitMix64. Shows the current counter
    value when use_counter=True; the seed is not displayed.
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

const char* model_run_docstr = R"""(
    Convenience instance method to start or resume model execution. Equivalent to `neworder.run(model)`.

    Returns:
        True if model succeeded, False otherwise
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
    Randomly changes categorical data, according to supplied transition probabilities, and returns the result
    as a new pandas Categorical - it does not modify series in-place, so the caller is responsible for
    assigning the result back, e.g. df[colname] = no.df.transition(model.mc, transition_matrix, df[colname]).
    The series must have a pandas "category" dtype (any category label type, e.g. strings, is supported) -
    convert it first with series = series.astype("category") if necessary. The row order of transition_matrix
    must correspond to the series' cat.categories order.
    Args:
        mc: The model's MonteCarlo engine (e.g. model.mc).
        transition_matrix: The probabilities of transitions between categories
        series: The pandas Series (categorical dtype) to transition
    Returns:
        The transitioned data, as a new pandas Categorical with the same categories/order as series.
)""";


const char* df_transition_conditional_docstr = R"""(
    Like transition(), but applies a different transition matrix per row depending on the corresponding value
    of another categorical column (group), e.g. transition probabilities that vary by age band or sex. It does
    not modify series in-place, so the caller is responsible for assigning the result back, e.g.
    df[colname] = no.df.transition_conditional(model.mc, matrices, df[groupname], df[colname]).
    Both series and group must have a pandas "category" dtype, and must be the same length and row-aligned -
    convert them first with series = series.astype("category") if necessary. matrices is a dict mapping each of
    group's category labels to the (square) transition matrix to apply to rows in that group; its row order
    must correspond to series' cat.categories order, as in transition(). Rows whose group value is NaN/missing
    are left untouched; every other category present in group.cat.categories must have a corresponding entry
    in matrices.
    Args:
        mc: The model's MonteCarlo engine (e.g. model.mc).
        matrices: dict mapping each category in group to the transition matrix (probabilities of transitions
            between categories) to apply where group has that value
        group: The pandas Series (categorical dtype) whose value selects the transition matrix per row
        series: The pandas Series (categorical dtype) to transition
    Returns:
        The transitioned data, as a new pandas Categorical with the same categories/order as series.
)""";

