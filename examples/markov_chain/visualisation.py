
from matplotlib import pyplot as plt

def show(model):
  # this seems to have a bug
  #model.summary.plot(kind='bar', width=1.0, stacked=True)
  dt = model.timeline().dt()
  plt.bar(model.summary.t, model.summary[0], width=dt)#, stacked=True)
  plt.bar(model.summary.t, model.summary[1], width=dt, bottom=model.summary[0])
  plt.bar(model.summary.t, model.summary[2], width=dt, bottom=model.summary[0]+model.summary[1])
  plt.legend(["State 0", "State 1", "State 2"])
  plt.title("State occupancy")
  plt.ylabel("Count")
  plt.xlabel("Time")

  #plt.savefig("docs/examples/img/markov_chain.png")
  plt.show()

