import neworder as no

from utils import assert_throws

#no.verbose()

def test_base() -> None:
  base = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

  assert_throws(RuntimeError, no.run, base) # RuntimeError: Tried to call pure virtual function "Model::step"

def test_multimodel() -> None:

  class TestModel(no.Model):
    def __init__(self) -> None:
      super().__init__(no.LinearTimeline(0,10,10), no.MonteCarlo.deterministic_identical_stream)

      self.x = 0.0

    def step(self) -> None:
      #no.log(self.mc.ustream(1))
      self.x += self.mc.ustream(1)

    def finalise(self) -> None:
      no.log(self.x)

  models = [TestModel(), TestModel()]

  [no.run(m) for m in models]

  assert models[0].x == models[1].x

if __name__ == "__main__":
  test_multimodel()