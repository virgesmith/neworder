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
    m = TestModel()


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
      super().__init__(no.LinearTimeline(0, 10, 10), no.MonteCarlo.deterministic_identical_stream)

      self.x = 0.0

    def step(self) -> None:
      self.x += self.mc.ustream(1)[0]

    def finalise(self) -> None:
      no.log(self.x)

  models = [TestModel(), TestModel()]

  [no.run(m) for m in models]

  assert models[0].x == models[1].x
