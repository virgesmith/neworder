"""
covid-19 case-based microsimulation
"""

import neworder

# 28 days later
neworder.timeline = neworder.Timeline(0, 28, [28])

npeople = 20

# Checks to run during the simulation
neworder.log_level = 1 # this doesnt do anything at the moment
neworder.do_checks = False

neworder.initialisations = {
  "model": { "module": "model", "class_": "Model", "args": (npeople) }
}

neworder.transitions = {
  "model": "model.step()"
}

neworder.checkpoints = {
  "model": "neworder.log(model.pop)"
}