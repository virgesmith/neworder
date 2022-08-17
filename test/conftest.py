
import pytest
import neworder as no

@pytest.fixture(scope="function")
def base_model():
  return no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

@pytest.fixture(scope="function")
def base_indep_model():
  return no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_independent_stream)

