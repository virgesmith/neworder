import warnings
from collections.abc import Callable, Iterable, Iterator

import pytest

import neworder as no


def test_base(base_model: no.Model) -> None:
    with pytest.raises(RuntimeError):
        no.run(base_model)  # RuntimeError: Tried to call pure virtual function "Model::step"


def test_base_not_initialised() -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            pass

    # there's no error until the instance is used: nanobind has no metaclass to check this on construction
    m = TestModel()
    with pytest.warns(RuntimeWarning, match="uninitialized"), pytest.raises(TypeError):
        no.run(m)


def test_step_not_implemented() -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.null())

    with pytest.raises(RuntimeError):
        no.run(TestModel())


def test_multimodel() -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.linear(0, end=10))
            self.mc = no.MonteCarlo(no.MonteCarlo.deterministic_identical_stream)
            self.x = 0.0

        def step(self) -> None:
            self.x += self.mc.ustream(1)[0]

        def finalise(self) -> None:
            no.log(self.x)

    models = [TestModel(), TestModel()]

    [no.run(m) for m in models]

    assert models[0].x == models[1].x


def test_runstate() -> None:
    no.verbose()

    class TestModel(no.Model):
        def __init__(self, *, do_halt: bool) -> None:
            super().__init__(no.timeline.null())
            self.do_halt = do_halt
            self.finalised = False

        def step(self) -> None:
            assert self.run_state == no.Model.RUNNING
            if self.do_halt:
                self.halt()
                assert self.run_state == no.Model.HALTED

        def finalise(self) -> None:
            assert self.run_state == no.Model.COMPLETED
            self.finalised = True

    no.verbose(False)

    # run without halting
    m = TestModel(do_halt=False)
    assert m.run_state == no.Model.NOT_STARTED
    no.run(m)
    assert m.run_state == no.Model.COMPLETED
    assert m.finalised

    # halt the run
    m = TestModel(do_halt=True)
    assert m.run_state == no.Model.NOT_STARTED
    no.run(m)
    assert m.run_state == no.Model.HALTED
    assert not m.finalised


def test_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.linear(0, end=2))

        def step(self) -> None:
            pass

    no.verbose()
    try:
        assert TestModel().run()
    finally:
        no.verbose(False)

    out = capsys.readouterr().out
    assert "starting TestModel model run" in out
    assert "TestModel.step()" in out
    assert "defaulted to no-op Model::finalise()" in out


def test_check_failure_as_error() -> None:
    class FailingModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.null())

        def step(self) -> None:
            pass

        def check(self) -> bool:
            return False

    # another test may have disabled checking
    no.checked()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="check\\(\\) FAILED"):
            no.run(FailingModel())


class _Raises(no.Model):
    def __init__(self, where: str, timeline: Iterable[no.Timestep] | None = None) -> None:
        super().__init__(no.timeline.linear(0, end=3) if timeline is None else timeline)
        self.where = where
        self.steps_taken = 0

    def modify(self) -> None:
        if self.where == "modify":
            raise ValueError(self.where)

    def step(self) -> None:
        self.steps_taken += 1
        if self.where == "step" and self.timestep.index == 2:
            raise ValueError(self.where)

    def check(self) -> bool:
        if self.where == "check" and self.timestep.index == 2:
            raise ValueError(self.where)
        return True

    def finalise(self) -> None:
        if self.where == "finalise":
            raise ValueError(self.where)


def _raising_timeline() -> Iterator[no.Timestep]:
    yield no.Timestep(1, 0, 1, 1.0)
    raise ValueError("timeline")


@pytest.mark.parametrize(
    ("model", "error"),
    [
        (lambda: _Raises("modify"), ValueError),
        (lambda: _Raises("step"), ValueError),
        (lambda: _Raises("check"), ValueError),
        (lambda: _Raises("finalise"), ValueError),
        (lambda: _Raises("timeline", _raising_timeline()), ValueError),
        (lambda: _Raises("timeline", [no.Timestep(1, 0, 1, 1.0), 1.0]), TypeError),  # ty:ignore[invalid-argument-type]
    ],
    ids=["modify", "step", "check", "finalise", "timeline", "not a timestep"],
)
def test_exception_fails_the_run(model: Callable[[], _Raises], error: type[Exception]) -> None:
    m = model()
    with pytest.raises(error):
        no.run(m)
    assert m.run_state == no.Model.FAILED
    steps_taken = m.steps_taken
    # rerunning would otherwise finalise a truncated run, or carry on past the failure
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        no.run(m)
    assert m.run_state == no.Model.FAILED
    assert m.steps_taken == steps_taken


def test_failed_check_fails_the_run() -> None:
    class FailsCheck(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.linear(0, end=5))
            self.steps_taken = 0

        def step(self) -> None:
            self.steps_taken += 1

        def check(self) -> bool:
            return self.timestep.index != 2

    m = FailsCheck()
    with pytest.warns(RuntimeWarning, match="check\\(\\) FAILED"):
        assert no.run(m) is False
    assert m.run_state == no.Model.FAILED
    assert m.steps_taken == 2
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        no.run(m)
    assert m.steps_taken == 2
