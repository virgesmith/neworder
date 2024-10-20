import importlib
import warnings

import pytest

import neworder as no

warnings.filterwarnings(action="ignore", category=RuntimeWarning, message=r"t=")


def test_basics() -> None:
    # just check you can read the attrs/call the functions
    assert hasattr(no, "verbose")
    assert hasattr(no, "checked")
    assert hasattr(no, "__version__")
    no.log("testing")
    no.log(1)
    no.log(no)
    no.log([1, 2, 3])
    no.log((1, 2, 3))
    no.log({1: 2, 3: 4})


def test_submodules() -> None:
    assert hasattr(no, "mpi")
    assert hasattr(no, "stats")
    assert hasattr(no, "df")


def test_dummy_model() -> None:
    class DummyModel(no.Model):
        def __init__(self) -> None:
            super().__init__(
                no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream
            )

        def step(self) -> None:
            pass

        def finalise(self) -> None:
            pass

    assert no.run(DummyModel())


@pytest.mark.filterwarnings("ignore:check()")
def test_check_flag() -> None:
    class FailingModel(no.Model):
        def __init__(self) -> None:
            super().__init__(
                no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream
            )

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
        assert no.mpi.RANK == no.mpi.COMM.Get_rank()
        assert no.mpi.SIZE == no.mpi.COMM.Get_size()
