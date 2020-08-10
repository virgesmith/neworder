
from matplotlib import pyplot as plt

import neworder
import seaborn as sns

class Graphics():
  def __init__(self):
    sns.set()

  def plot(self, model):
    model.summary = model.summary.fillna(0)
    model.summary.index = range(1,len(model.summary)+1)
    # force ordering for stacked bar chart
    plt.plot(range(model.timeline().nsteps()+1), model.infection_rate)
    plt.plot(range(model.timeline().nsteps()+1), model.mortality_rate)
    #plt.plot(range(1,neworder.timeline.nsteps()+1), self.summary[State.DECEASED])

    model.summary.plot(kind='bar', width=1.0, stacked=True)

    # # neworder.log(self.summary.tail(1)[State.DECEASED].values[0] / self.npeople * 100.0)
    # # neworder.log("Overall mortality: %f%%" % (self.summary.tail(1)[State.DECEASED].values[0] / self.npeople * 100.0))
    plt.show()
    #self.pop.to_csv("pop.csv", index=False)

