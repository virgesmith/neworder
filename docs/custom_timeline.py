import neworder


class AsymptoticTimeline(neworder.Timeline):
  def __init__(self) -> None:
    super().__init__()
    self.t = 1.0
    self.i = 0

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
  def index(self) -> int:
    return self.i

  @property
  def time(self) -> float:
    return 1.0 - self.t

  @property
  def dt(self) -> float:
    return self.t / 2

  def next(self) -> None:
    self.i += 1
    self.t /= 2

  @property
  def at_end(self) -> bool:
    return False

t = CustomTimeline()

while t.time < 0.99:
  print(t.time)
  t.next()

