import pytest
import numpy as np
import neworder as no

from utils import assert_throws

no.verbose()

def test_base():
  base = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  assert_throws(NotImplementedError, no.run, base)

def test_multimodel():
  pass
  # TODO ensure 2 models can work...

