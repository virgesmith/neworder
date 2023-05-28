import neworder


class AsymptoticTimeline(neworder.Timeline):
  def __init__(self) -> None:
    super().__init__()
    self.t = 1.0

  @property
  def start(self) -> float:
    return 0.0

  @property
  def end(self) -> float:
    return 1.0

  @property
  def nsteps(self) -> int:
    return -1

  @property
  def time(self) -> float:
    return 1.0 - self.t

  @property
  def dt(self) -> float:
    return self.t / 2

  def _next(self) -> None:
    self.t /= 2

  @property
  def at_end(self) -> bool:
    return False


class Model(neworder.Model):
  def __init__(self) -> None:
    super().__init__(AsymptoticTimeline())

  def step(self) -> None:
    neworder.log(f"{self.timeline.index}: {self.timeline.time}")
    if self.timeline.time > 0.99:
      self.halt()

neworder.run(Model())
