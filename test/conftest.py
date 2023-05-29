
import pytest
import neworder as no

@pytest.fixture(scope="function")
def base_model() -> no.Model:
  return no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

@pytest.fixture(scope="function")
def base_indep_model() -> no.Model:
  return no.Model(no.NoTimeline())

