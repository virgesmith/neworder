import numpy as np
from matplotlib import pyplot as plt
import neworder as no

def plot(model):
  plot_age(model)
  plt.show()
  plot_parity(model)
  plt.show()

def plot_age(model):
  bins = np.arange(model.max_rate_age)

  b = [model.population.time_of_baby_1,
       model.population.time_of_baby_2,
       model.population.time_of_baby_3,
       model.population.time_of_baby_4,
       model.population.time_of_baby_5]
  plt.hist(b, bins, stacked=True)
  plt.hist(model.population.time_of_death, bins, color='black')
  plt.title("Competing risks of childbirth and death")
  plt.legend(["1st birth", "2nd birth", "3rd birth", "4th birth", "5th birth", "Death"])
  plt.xlabel("Age (y)")
  plt.ylabel("Frequency")
  #plt.savefig("./docs/examples/img/competing_hist_100k.png", dpi=80)

def plot_parity(model):
  bins = np.arange(model.population.parity.max())-0.25
  plt.hist(model.population.parity, bins, width=0.5)
  plt.title("Births during lifetime")
  plt.xlabel("Number of children")
  plt.ylabel("Frequency")
  #plt.savefig("./docs/examples/img/competing_births_100k.png", dpi=80)
