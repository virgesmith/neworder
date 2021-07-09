import pytest
import numpy as np
import neworder as no

from utils import assert_throws

#no.verbose()

def test_base():
  base = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  assert_throws(NotImplementedError, no.run, base)

def test_multimodel():
  
  class TestModel(no.Model):
    def __init__(self):
      super().__init__(no.LinearTimeline(0,10,10), no.MonteCarlo.deterministic_identical_stream)

      self.x = 0.0

    def step(self):
      #no.log(self.mc.ustream(1))
      self.x += self.mc.ustream(1)

    def finalise(self):
      no.log(self.x)

  models = [TestModel(), TestModel()]

  [no.run(m) for m in models]

  assert models[0].x == models[1].x

if __name__ == "__main__":
  test_multimodel()