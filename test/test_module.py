import importlib.util
import warnings

import pytest

import neworder as no

warnings.filterwarnings(action="ignore", category=RuntimeWarning, message=r"t=")


def test_basics() -> None:
    # just check you can read the attrs/call the functions
    assert hasattr(no, "verbose")
    assert hasattr(no, "checked")
    assert hasattr(no, "__version__")
    assert isinstance(no.freethreaded(), bool)
    no.log("testing")
    no.log(1)
    no.log(no)
    no.log([1, 2, 3])
    no.log((1, 2, 3))
    no.log({1: 2, 3: 4})


def test_log_text(capsys: pytest.CaptureFixture[str]) -> None:
    no.log("a\x00b", "c")
    assert capsys.readouterr().out.endswith("] a\x00b c\n")
    # capsys's stdout is strict UTF-8, so a lone surrogate raises just as it would with print
    with pytest.raises(UnicodeEncodeError):
        print("x\udc80y")
    with pytest.raises(UnicodeEncodeError):
        no.log("x\udc80y")


def test_submodules() -> None:
    assert hasattr(no, "mpi")
    assert hasattr(no, "stats")
    assert hasattr(no, "df")


def test_dummy_model() -> None:
    class DummyModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.null(), no.MonteCarlo.deterministic_identical_stream)

        def step(self) -> None:
            pass

        def finalise(self) -> None:
            pass

    assert no.run(DummyModel())


@pytest.mark.filterwarnings("ignore:check()")
def test_check_flag() -> None:
    class FailingModel(no.Model):
        def __init__(self) -> None:
            super().__init__(no.timeline.null(), no.MonteCarlo.deterministic_identical_stream)

        def step(self) -> None:
            pass

        def check(self) -> bool:
            return False

    # fails
    assert not no.run(FailingModel())

    no.checked(False)
    # succeeds
    assert no.run(FailingModel())


def test_mpi() -> None:
    # if no mpi4py, assume serial like module does
    if importlib.util.find_spec("mpi4py") is None:
        assert not no.mpi.COMM
        assert no.mpi.RANK == 0
        assert no.mpi.SIZE == 1
    else:
        assert no.mpi.COMM
        assert no.mpi.COMM.Get_rank() == no.mpi.RANK
        assert no.mpi.COMM.Get_size() == no.mpi.SIZE
