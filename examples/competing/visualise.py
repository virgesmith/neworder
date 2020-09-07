import numpy as np
from matplotlib import pyplot as plt
import neworder as no

def plot(model):
  buckets = range(model.max_rate_age)

  b = [ model.population.TimeOfBaby1[~np.isnan(model.population.TimeOfBaby1)],
        model.population.TimeOfBaby2[~np.isnan(model.population.TimeOfBaby2)],
        model.population.TimeOfBaby3[~np.isnan(model.population.TimeOfBaby3)],
        model.population.TimeOfBaby4[~np.isnan(model.population.TimeOfBaby4)] ]
  plt.hist(b, buckets, stacked=True)
  plt.hist(model.population.TimeOfDeath, buckets, color='black')
  plt.title("Competing risks of childbirth and death")
  plt.legend(["Birth 1", "Birth 2", "Birth 3", "Birth 4", "Death"])
  plt.xlabel("Age (y)")
  plt.ylabel("Frequency")
  #plt.savefig("./docs/examples/img/competing_hist_100k.png")
  plt.show()

def stats(model):
  # # compute mean
  no.log("birth rate = %f" % np.mean(model.population.Parity))
  no.log("pct mother = %f" % (100.0 * np.mean(model.population.Parity > 0)))
  no.log("life exp. = %f" % np.mean(model.population.TimeOfDeath))
  #return True
