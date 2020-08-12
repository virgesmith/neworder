"""
Markov chain continuous vs discrete
"""

import neworder

days = 28
dt = 0.1
neworder.timeline = neworder.Timeline(0, days, [int(days/dt)])

npeople = 10000

# Checks to run during the simulation
neworder.log_level = 1 # this doesnt do anything at the moment
neworder.do_checks = False

neworder.initialisations = {
  "model": { "module": "model", "class_": "MarkovChain", "args": (npeople,) }
}

neworder.dataframe.transitions = {
  "d": "model.step()"
}

neworder.checkpoints = {
  "model": "model.plot()"
}
