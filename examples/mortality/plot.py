
import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns
import animation

def plot(population, population2, filename=None, anim_filename=None):
  #sns.set()
  y1, x1 = np.histogram(population2.TimeOfDeath, int(max(population.Age)))
  plt.plot(x1[1:], y1)
  y2, x2 = np.histogram(population.TimeOfDeath, int(max(population.Age)))
  plt.plot(x2[1:], y2)
  plt.title("Mortality model sampling algorithm comparison")
  plt.legend(["Continuous", "Discrete"])

  if filename is not None:
    plt.savefig(filename)

  h = animation.Hist(population2.TimeOfDeath, int(max(population.Age)))
  if anim_filename is not None:
    h.save(anim_filename)
  h.show()
