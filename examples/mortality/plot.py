
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Hist:
  def __init__(self, data, numbins):

    fig, ax = plt.subplots()
    # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
    self.n, _bins, self.patches = plt.hist(data, numbins, facecolor='black')

    ax.set_title("Discrete case-based mortality model (%d people)" % len(data))
    ax.set_xlabel("Age at Death")
    ax.set_ylabel("Persons")

    self.anim = animation.FuncAnimation(fig, self.__animate, interval=100, frames=numbins, repeat=True, repeat_delay=3000)

  def save(self, filename):
    self.anim.save(filename, dpi=80, writer='imagemagick')

  def show(self):
    plt.show()

  def __animate(self, frameno):
    i = 0
    for rect, h in zip(self.patches, self.n):
      rect.set_height(h if i <= frameno else 0)
      i = i + 1
    return self.patches


def plot(pop_disc, pop_cont, filename=None, anim_filename=None):
  #sns.set()
  y1, x1 = np.histogram(pop_disc.age_at_death, int(max(pop_disc.age)))
  plt.plot(x1[1:], y1)
  y2, x2 = np.histogram(pop_cont.age_at_death, int(max(pop_disc.age)))
  plt.plot(x2[1:], y2)
  plt.title("Mortality model sampling algorithm comparison")
  plt.legend(["Discrete", "Continuous"])

  if filename is not None:
    plt.savefig(filename)

  h = Hist(pop_disc.age_at_death, int(max(pop_disc.age)))
  if anim_filename is not None:
    h.save(anim_filename)
  h.show()
