import warnings

import pytest

import neworder as no


def test_base(base_model: no.Model) -> None:
    with pytest.raises(RuntimeError):
        no.run(base_model)  # RuntimeError: Tried to call pure virtual function "Model::step"


def test_base_not_initialised() -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            pass

    with pytest.raises(TypeError):
        _ = TestModel()


def test_step_not_implemented() -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline())

    with pytest.raises(RuntimeError):
        no.run(TestModel())


def test_default_seeder() -> None:
    class DefaultModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline())
            self.x = self.mc.raw()

    class ExplicitModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline(), no.MonteCarlo.deterministic_independent_stream)
            self.x = self.mc.raw()

    class DifferentModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.NoTimeline(), lambda: 42)
            self.x = self.mc.raw()

    assert DefaultModel().x == ExplicitModel().x
    assert DefaultModel().x != DifferentModel().x


def test_multimodel() -> None:
    class TestModel(no.Model):
        def __init__(self) -> None:
            super().__init__(
                no.LinearTimeline(0, 10, 10),
                no.MonteCarlo.deterministic_identical_stream,
            )
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
            super().__init__(no.NoTimeline())
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
            super().__init__(no.LinearTimeline(0, 2, 2))

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
            super().__init__(no.NoTimeline())

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
