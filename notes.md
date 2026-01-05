# GIL-free/multithreaded notes

- use threading.Lock as a context manager to release lock on exception rather than explicit acquire/release
- thread_id is set on first use (e.g. a call to `neworder.log` or the C++ internal equivalent)
- results are not necessarily deterministic **even with deterministic seeds** as the order of thread execution is potentially variable (e.g. unique_index)
- better to seed RNGs with an explicit value rather than `thread_id()`
- inter-thread synchronisation/communication is not possible/trickier than MPI (c.f. point above re: nondeterministic execution)
- ThreadPoolExecutor will potentially reuse threads, so you may see the same thread_id in supposedly different threads

- Why doesn't the (absence of) GIL have a greater impact? In the option example, execution times are not quite as expected:
  - sequential: ~3.0s
  - parallel, with GIL: ~1.1s
  - parallel, without GIL: ~0.9s
