import neworder as no

import pytest


def test_base(base_model: no.Model) -> None:
  with pytest.raises(RuntimeError):
    no.run(base_model)  # RuntimeError: Tried to call pure virtual function "Model::step"


def test_multimodel() -> None:

  class TestModel(no.Model):
    def __init__(self) -> None:
      super().__init__(no.LinearTimeline(0, 10, 10), no.MonteCarlo.deterministic_identical_stream)

      self.x = 0.0

    def step(self) -> None:
      self.x += self.mc.ustream(1)[0]

    def finalise(self) -> None:
      no.log(self.x)

  models = [TestModel(), TestModel()]

  [no.run(m) for m in models]

  assert models[0].x == models[1].x
