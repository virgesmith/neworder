import pytest

import neworder as no


@pytest.fixture(scope="function")
def base_model() -> no.Model:
    return no.Model(no.timeline.null())


@pytest.fixture(scope="function")
def mc() -> no.MonteCarlo:
    return no.MonteCarlo(no.MonteCarlo.deterministic_identical_stream)


@pytest.fixture(scope="function")
def indep_mc() -> no.MonteCarlo:
    return no.MonteCarlo()
