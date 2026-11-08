"""
Pins the behaviour of the C++/Python boundary that the framework and its users rely on, and the values of the random
streams, independently of the library that implements the bindings.
"""

import enum
import gc
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import neworder as no


@pytest.fixture
def mc() -> no.MonteCarlo:
    # the python reference to mc keeps its model alive
    return no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream).mc


class _CheckModel(no.Model):
    def __init__(self, check_result: Any) -> None:
        super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
        self.check_result = check_result

    def step(self) -> None:
        pass

    def check(self) -> bool:
        return self.check_result


class _NumpyBoolTimeline(no.Timeline):
    """A custom timeline whose properties return numpy scalars, as computed properties naturally do"""

    def __init__(self) -> None:
        super().__init__()
        self.t = np.float64(0.0)

    @property
    def start(self) -> float:
        return 0.0

    @property
    def end(self) -> float:
        return 2.0

    @property
    def time(self) -> float:
        return float(self.t)

    @property
    def dt(self) -> float:
        return 1.0

    def _next(self) -> None:
        self.t += 1.0

    @property
    def at_end(self) -> bool:
        return self.t >= 2.0  # ty: ignore[invalid-return-type]


@pytest.mark.parametrize(
    "p",
    [
        [0.5, 0.5, 0.5],
        (0.5, 0.5, 0.5),
        np.array([0.5, 0.5, 0.5]),
        pd.Series([0.5, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
    ],
    ids=["list", "tuple", "ndarray", "series", "float32"],
)
def test_array_args_accept_array_likes(mc: no.MonteCarlo, p: Any) -> None:
    result = mc.hazard(p)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (3,)


def test_array_args_accept_integers(mc: no.MonteCarlo) -> None:
    assert (mc.hazard(np.array([0, 1])) == [0.0, 1.0]).all()


def test_array_args_accept_readonly(mc: no.MonteCarlo) -> None:
    p = np.array([0.5, 0.5])
    p.flags.writeable = False
    assert mc.hazard(p).shape == (2,)
    assert mc.stopping(p).shape == (2,)
    assert not no.time.isnever(p).any()


def test_array_args_preserve_shape(mc: no.MonteCarlo) -> None:
    assert mc.hazard(np.full((2, 3), 0.5)).shape == (2, 3)
    assert no.time.isnever(np.zeros((2, 3))).shape == (2, 3)


def test_array_args_do_not_modify_input(mc: no.MonteCarlo) -> None:
    p = np.array([0.25, 0.75])
    mc.hazard(p)
    assert (p == [0.25, 0.75]).all()


def test_array_args_reject_non_numeric(mc: no.MonteCarlo) -> None:
    with pytest.raises(TypeError):
        mc.hazard("abc")


def test_isnever_overloads() -> None:
    # all scalar types must resolve to the scalar overload
    for t in (1.0, 1, np.float64(1.0), np.float32(1.0)):
        assert isinstance(no.time.isnever(t), bool)
    assert no.time.isnever(no.time.NEVER) is True

    for a in ([1.0, no.time.NEVER], (1.0, no.time.NEVER), np.array([1.0, no.time.NEVER]), pd.Series([1.0, np.nan])):
        result = no.time.isnever(a)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.bool_
        assert result.tolist() == [False, True]


def test_returned_arrays(mc: no.MonteCarlo) -> None:
    for a in (mc.ustream(3), mc.hazard(0.5, 3), mc.stopping(0.5, 3), mc.first_arrival([1.0, 0.0], 1.0, 3)):
        assert a.dtype == np.float64
        assert a.shape == (3,)
        assert a.flags.writeable
        assert a.flags.c_contiguous
        assert a.flags.owndata
    for a in (mc.sample(3, np.array([0.5, 0.5])), mc.counts([1.0, 1.0, 1.0], 1.0)):
        assert a.dtype == np.int64
        assert a.shape == (3,)
        assert a.flags.writeable
    # n=0 so as not to advance the global index that other tests depend on
    assert no.df.unique_index(0).dtype == np.int64


def test_returned_arrays_outlive_model() -> None:
    a = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream).mc.ustream(1000)
    gc.collect()
    assert ((a >= 0.0) & (a < 1.0)).all()


def test_array_sizes(mc: no.MonteCarlo) -> None:
    assert mc.ustream(0).shape == (0,)
    assert mc.ustream(np.int64(2)).shape == (2,)
    assert mc.sample(np.prod((2, 2)), np.array([0.5, 0.5])).shape == (4,)
    for f in (mc.ustream, lambda n: mc.hazard(0.5, n), lambda n: mc.stopping(0.5, n)):
        with pytest.raises(ValueError):
            f(-1)
    with pytest.raises(TypeError):
        mc.ustream(2.0)


def test_default_seeder() -> None:
    assert no.Model(no.NoTimeline()).mc.seed() == 19937 + no.mpi.RANK


def test_seeder_numpy_int() -> None:
    assert no.Model(no.NoTimeline(), lambda: np.int64(3)).mc.seed() == 3
    no.SplitMix64(lambda: np.int64(3)).uarray(1)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("result", [True, 1, np.True_], ids=["bool", "int", "numpy_bool"])
def test_check_result_truthy(result: Any) -> None:
    assert no.run(_CheckModel(result)) is True


@pytest.mark.parametrize("result", [False, 0, np.False_], ids=["bool", "int", "numpy_bool"])
def test_check_result_falsy(result: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert no.run(_CheckModel(result)) is False


def test_timeline_at_end_numpy_bool() -> None:
    class Model(no.Model):
        def __init__(self) -> None:
            super().__init__(_NumpyBoolTimeline(), no.MonteCarlo.deterministic_identical_stream)
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    m = Model()
    assert no.run(m)
    assert m.steps == 2


def test_python_overrides_can_call_base() -> None:
    class Model(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
            self.called: list[str] = []

        def modify(self) -> None:
            super().modify()
            self.called.append("modify")

        def step(self) -> None:
            self.called.append("step")

        def check(self) -> bool:
            self.called.append("check")
            return super().check()

        def finalise(self) -> None:
            super().finalise()
            self.called.append("finalise")

    m = Model()
    assert no.run(m)
    assert m.called == ["modify", "step", "check", "finalise"]


def test_python_overrides_called_from_python_run_once() -> None:
    class Model(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
            self.called: list[str] = []

        def modify(self) -> None:
            self.called.append("modify")
            super().modify()

        def step(self) -> None:
            self.called.append("step")
            super().step()

        def check(self) -> bool:
            self.called.append("check")
            return super().check()

        def finalise(self) -> None:
            self.called.append("finalise")
            super().finalise()

    m = Model()
    m.modify()
    assert m.check()
    m.finalise()
    with pytest.raises(RuntimeError):
        m.step()
    assert m.called == ["modify", "check", "finalise", "step"]


def test_python_override_calling_pure_base_raises() -> None:
    class Model(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
            self.steps = 0

        def step(self) -> None:
            self.steps += 1
            super().step()

    m = Model()
    with pytest.raises(RuntimeError):
        no.run(m)
    assert m.steps == 1


def test_instance_assigned_override_needs_class_override() -> None:
    # nanobind (unlike pybind11) first checks the class for an override, and only then fetches the method from the
    # instance, so a method assigned to an instance is ignored by C++ unless the class also overrides it
    class StepOnly(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)
            self.called: list[str] = []

        def step(self) -> None:
            self.called.append("step")

    class WithFinalise(StepOnly):
        def finalise(self) -> None:
            self.called.append("class finalise")

    for cls, expected in ((StepOnly, ["step"]), (WithFinalise, ["step", "instance finalise"])):
        m = cls()
        m.finalise = lambda m=m: m.called.append("instance finalise")
        assert no.run(m)
        assert m.called == expected


def test_timeline_index_must_be_int() -> None:
    class StrIndexTimeline(_NumpyBoolTimeline):
        @property
        def index(self) -> int:
            return "zero"  # ty: ignore[invalid-return-type]

    class M(no.Model):
        def __init__(self) -> None:
            super().__init__(StrIndexTimeline(), no.MonteCarlo.deterministic_identical_stream)

        def step(self) -> None:
            pass

    with pytest.raises(TypeError, match=r"StrIndexTimeline\.index must be an int64, got 'zero'"):
        no.run(M())


def test_timeline_repr() -> None:
    class Timeline(_NumpyBoolTimeline):
        calls = 0

        def __repr__(self) -> str:
            Timeline.calls += 1
            return "custom " + super().__repr__()

    assert repr(_NumpyBoolTimeline()) == "<_NumpyBoolTimeline index=0>"
    assert repr(Timeline()) == "custom <Timeline index=0>"
    assert Timeline.calls == 1
    # the C++ implementations are still used, and inherited by any python subclasses
    assert repr(no.NoTimeline()) == "<neworder.NoTimeline stepped=False>"
    assert repr(no.LinearTimeline(0.0, 1.0, 1)).startswith("<neworder.LinearTimeline")
    assert repr(no.NumericTimeline([0.0, 1.0])).startswith("<neworder.NumericTimeline")

    # a python __repr__ is seen by C++, which logs it when the model is constructed
    no.verbose(True)
    try:
        calls = Timeline.calls
        no.Model(Timeline(), no.MonteCarlo.deterministic_identical_stream)
        assert Timeline.calls == calls + 1
    finally:
        no.verbose(False)


def test_object_identity() -> None:
    timeline = _NumpyBoolTimeline()
    m = no.Model(timeline, no.MonteCarlo.deterministic_identical_stream)
    assert m.timeline is timeline
    assert m.mc is m.mc


def test_timeline_downcast() -> None:
    m = no.Model(no.LinearTimeline(0.0, 1.0, 1), no.MonteCarlo.deterministic_identical_stream)
    assert isinstance(m.timeline, no.LinearTimeline)


def test_timeline_kept_alive_by_model() -> None:
    m = no.Model(_NumpyBoolTimeline(), no.MonteCarlo.deterministic_identical_stream)
    gc.collect()
    assert m.timeline.time == 0.0


def test_mc_keeps_model_alive() -> None:
    mc = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream).mc
    gc.collect()
    assert mc.ustream(1).shape == (1,)


def test_run_state() -> None:
    assert issubclass(no.Model.RunState, enum.Enum)
    assert no.Model.NOT_STARTED is no.Model.RunState.NOT_STARTED
    m = _CheckModel(True)
    assert m.run_state is no.Model.RunState.NOT_STARTED
    no.run(m)
    assert m.run_state is no.Model.RunState.COMPLETED


def test_timeline_arguments() -> None:
    assert no.NumericTimeline(np.array([0.0, 1.0])).dt == 1.0  # ty: ignore[invalid-argument-type]
    assert no.NumericTimeline((0.0, 1.0)).dt == 1.0
    assert no.LinearTimeline(0, 1, np.int64(2)).dt == 0.5
    with pytest.raises(TypeError):
        no.LinearTimeline(0, 1, 2.0)


def test_keyword_arguments(mc: no.MonteCarlo) -> None:
    lambda_ = np.array([1.0, 0.0])
    assert mc.ustream(n=1).shape == (1,)
    assert mc.hazard(p=0.5, n=1).shape == (1,)
    assert mc.hazard(p=np.array([0.5])).shape == (1,)
    assert mc.stopping(lambda_=0.5, n=1).shape == (1,)
    assert mc.first_arrival(lambda_=lambda_, dt=1.0, n=1).shape == (1,)
    assert mc.next_arrival(startingpoints=np.zeros(1), lambda_=lambda_, dt=1.0).shape == (1,)
    assert mc.next_arrival(np.zeros(1), lambda_, 1.0, relative=True, minsep=0.5).shape == (1,)
    assert no.time.isnever(t=1.0) is False


def test_splitmix_arguments() -> None:
    sm = no.SplitMix64(lambda: 1)
    assert sm.uarray(1, 2).shape == ()
    assert sm.raw(1, 2).shape == ()
    assert sm.uarray(1, 2).dtype == np.float64
    assert sm.raw(1, 2).dtype == np.int64
    assert sm.uarray(np.array([1, 2]), 3).shape == (2,)
    assert sm.uarray(np.array([1, 2]), np.array([3, 4, 5])).shape == (2, 3)
    assert (sm.raw([1, 2]) == sm.raw(np.array([1, 2]))).all()
    # floating point arrays are truncated
    assert (sm.raw(np.array([1.5, 2.5])) == sm.raw(np.array([1, 2]))).all()
    for invalid in (np.int64(3), "abc", np.zeros((2, 2), dtype=np.int64)):
        with pytest.raises(TypeError):
            sm.uarray(invalid)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        sm.uarray()


def test_splitmix_counter_keyword_only() -> None:
    assert "counter" in repr(no.SplitMix64(lambda: 1, use_counter=True))
    with pytest.raises(TypeError):
        no.SplitMix64(lambda: 1, True)  # ty: ignore[too-many-positional-arguments]


def test_transition_arguments(mc: no.MonteCarlo) -> None:
    series = pd.Series(pd.Categorical(["a", "b", "a", "b", np.nan], categories=["a", "b"]))
    codes = series.cat.codes.copy()
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = ["b", "a", "b", "a", None]

    for m in (matrix, matrix.tolist(), np.asfortranarray(matrix)):
        result = no.df.transition(mc, m, series)
        assert isinstance(result, pd.Categorical)
        assert [None if pd.isna(x) else x for x in result] == expected
    # the input isn't modified
    assert (series.cat.codes == codes).all()

    group = pd.Series(pd.Categorical(["x", "y", "x", "y", "x"]))
    result = no.df.transition_conditional(mc, {"x": matrix.tolist(), "y": matrix}, group, series)
    assert [None if pd.isna(x) else x for x in result] == expected


def test_log(capsys: pytest.CaptureFixture[str]) -> None:
    no.log("a", 1, [2])
    out = capsys.readouterr().out
    assert re.fullmatch(rf"\[py {no.mpi.RANK}/{no.mpi.SIZE}\(\d+\)\] a 1 \[2\]\n", out)


_SERIES = pd.Series(pd.Categorical(["a", "b", "c", "a", "b", "c"]))
_GROUP = pd.Series(pd.Categorical(["x", "y", "x", "y", "x", "y"]))
_MATRIX = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])


def _reset(mc: no.MonteCarlo) -> Any:
    mc.ustream(2)
    mc.reset()
    return mc.ustream(2)


MC_CASES: dict[str, Callable[[no.MonteCarlo], Any]] = {
    "ustream": lambda mc: mc.ustream(4),
    "raw": lambda mc: np.array([mc.raw(), mc.raw()], dtype=np.uint64),
    "sample": lambda mc: mc.sample(6, np.array([0.2, 0.3, 0.5])),
    "hazard_scalar": lambda mc: mc.hazard(0.5, 6),
    "hazard_array": lambda mc: mc.hazard(np.array([0.1, 0.5, 0.9, 0.5])),
    "stopping_scalar": lambda mc: mc.stopping(0.25, 3),
    "stopping_array": lambda mc: mc.stopping(np.array([0.1, 0.5, 0.9])),
    "counts": lambda mc: mc.counts(np.array([1.0, 2.0, 3.0]), 1.0),
    "arrivals": lambda mc: mc.arrivals(np.array([1.0, 2.0, 3.0, 0.0]), 1.0, 2, 0.5),
    "first_arrival": lambda mc: mc.first_arrival(np.array([0.5, 1.0, 0.0]), 1.0, 3, 0.5),
    "next_arrival": lambda mc: mc.next_arrival(np.array([0.0, 1.0, no.time.NEVER]), np.array([0.5, 1.0, 0.0]), 1.0),
    "next_arrival_relative": lambda mc: mc.next_arrival(
        np.array([0.0, 1.0, no.time.NEVER]), np.array([0.5, 1.0, 0.0]), 1.0, relative=True, minsep=0.25
    ),
    "as_np": lambda mc: no.as_np(mc).random(3),
    "reset": _reset,
    "transition": lambda mc: no.df.transition(mc, _MATRIX, _SERIES).codes,
    "transition_conditional": lambda mc: (
        no.df.transition_conditional(mc, {"x": _MATRIX, "y": _MATRIX[[1, 2, 0]]}, _GROUP, _SERIES).codes
    ),
}

SPLITMIX_CASES: dict[str, Callable[[no.SplitMix64], Any]] = {
    "uarray_scalars": lambda sm: sm.uarray(1, 2),
    "uarray_1d": lambda sm: sm.uarray(np.array([1, 2, 3]), 4),
    "uarray_2d": lambda sm: sm.uarray(np.array([1, 2]), 3, np.array([4, 5])),
    "raw_scalars": lambda sm: sm.raw(1, 2),
    "raw_1d": lambda sm: sm.raw(np.array([1, 2, 3]), 4),
}

# Generated from the pybind11 build at the point these tests were introduced. These must not change: a change means
# that existing models no longer reproduce their results.
MC_EXPECTED: dict[str, Any] = {
    "ustream": [0.33778882725164294, 0.04767065867781639, 0.8131122114136815, 0.24954832065850496],
    "raw": [6231104047474287856, 14999272868227999252],
    "sample": [1, 0, 2, 1, 1, 2],
    "hazard_scalar": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    "hazard_array": [0.0, 1.0, 1.0, 1.0],
    "stopping_scalar": [4.34133740077988, 12.173756769645559, 0.8275446301574549],
    "stopping_array": [10.853343501949698, 6.086878384822779, 0.22987350837707082],
    "counts": [0, 0, 4],
    "arrivals": [[0.36177811673165666, 0.930740169244778], [0.9318538353164916, 1.8495370491398397]],
    "first_arrival": [1.58533435019497, 0.7068861575393637, 1.5830648848843396],
    "next_arrival": [1.08533435019497, 1.2068861575393637, float("nan")],
    "next_arrival_relative": [1.33533435019497, 1.4568861575393637, float("nan")],
    "as_np": [0.33778882725164294, 0.04767065867781639, 0.8131122114136815],
    "reset": [0.33778882725164294, 0.04767065867781639],
    "transition": [0, 1, 2, 0, 1, 2],
    "transition_conditional": [0, 0, 2, 1, 1, 1],
}
SPLITMIX_EXPECTED: dict[str, Any] = {
    "uarray_scalars": 0.7227816673726384,
    "uarray_1d": [0.0757271816423265, 0.3749668037798326, 0.5716265527680476],
    "uarray_2d": [[0.19050353437107703, 0.05054849667248662], [0.6347573569818676, 0.061749276716171475]],
    "raw_scalars": -5113775634517424828,
    "raw_1d": [1396919939179314781, 6916916665463440878, -7902095349060547300],
}


def _assert_matches(actual: Any, expected: Any) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=actual.dtype)
    assert actual.shape == expected.shape
    if np.issubdtype(actual.dtype, np.floating):
        # libm's log() may differ in the last few bits across platforms
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0, equal_nan=True)
    else:
        assert (actual == expected).all()


@pytest.mark.parametrize("case", MC_CASES)
def test_mc_streams(mc: no.MonteCarlo, case: str) -> None:
    _assert_matches(MC_CASES[case](mc), MC_EXPECTED[case])


@pytest.mark.parametrize("case", SPLITMIX_CASES)
def test_splitmix_streams(case: str) -> None:
    _assert_matches(SPLITMIX_CASES[case](no.SplitMix64(lambda: 19937)), SPLITMIX_EXPECTED[case])
