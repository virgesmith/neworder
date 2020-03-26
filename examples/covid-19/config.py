"""
covid-19 case-based microsimulation
"""

import neworder

days = 180
dt = 1
neworder.timeline = neworder.Timeline(0, days, [int(days/dt)])

npeople = 10000

# Checks to run during the simulation
neworder.log_level = 1 # this doesnt do anything at the moment
neworder.do_checks = False

neworder.initialisations = {
  "model": { "module": "model", "class_": "Model", "args": (npeople,) }
  #"model": { "module": "model2", "class_": "Model", "args": (npeople,) }
}

neworder.transitions = {
  "model": "model.step()"
}

neworder.checkpoints = {
  "model": "model.plot()"
}