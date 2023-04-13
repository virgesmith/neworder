from typing import Optional, Union
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.animation as animation  # type: ignore


class Hist:
  def __init__(self, data: pd.DataFrame, numbins: int) -> None:

    fig, ax = plt.subplots()
    # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
    self.n, _bins, self.patches = plt.hist(data, numbins, facecolor='black')

    ax.set_title("Discrete case-based mortality model (%d people)" % len(data))
    ax.set_xlabel("Age at Death")
    ax.set_ylabel("Persons")

    self.anim = animation.FuncAnimation(fig, self.__animate, interval=100, frames=numbins, repeat=False)

  def save(self, filename: str) -> None:
    # there seems to be no way of preventing passing the loop once setting to the saved gif and it loops forever, which is very annoying
    self.anim.save(filename, dpi=80, writer=animation.ImageMagickWriter(extra_args=["-loop", "1"]))

  def show(self) -> None:
    plt.show()

  def __animate(self, frameno: int) -> Union[list, list[list]]:
    i = 0
    for rect, h in zip(self.patches, self.n):
      rect.set_height(h if i <= frameno else 0)
      i = i + 1
    return self.patches


def plot(pop_disc: pd.DataFrame, pop_cont: pd.DataFrame, filename: Optional[str]=None, anim_filename: Optional[str]=None) -> None:
  bins = int(max(pop_disc.age_at_death.max(), pop_cont.age_at_death.max())) + 1
  rng = (0.0, float(bins))
  y1, x1 = np.histogram(pop_disc.age_at_death, bins, range=rng)
  plt.plot(x1[1:], y1)
  y2, x2 = np.histogram(pop_cont.age_at_death, bins, range=rng)
  plt.plot(x2[1:], y2)
  plt.title("Mortality model sampling algorithm comparison")
  plt.legend(["Discrete", "Continuous"])
  plt.xlabel("Age at Death")
  plt.ylabel("Persons")

  if filename is not None:
    plt.savefig(filename, dpi=80)

  h = Hist(pop_disc.age_at_death, int(max(pop_disc.age)))
  if anim_filename is not None:
    h.save(anim_filename)
  h.show()
