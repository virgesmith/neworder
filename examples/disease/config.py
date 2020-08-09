"""
Infectious disease case-based microsimulation
"""

import neworder

days = 180
dt = 1
timeline = neworder.Timeline(0, days, [int(days/dt)])

npeople = 10000

# Checks to run during the simulation
#neworder.log_level = 1 # this doesnt do anything at the moment

from model import DiseaseModel
from graphics import Graphics

  # "graphics": Graphics() 

transitions = {
  "model": "neworder.model.step()"
}

checkpoints = {
  "finalise": "neworder.model.finalise()",
  "graphics": "Graphics().plot(neworder.model)"
}

neworder.model = DiseaseModel(timeline, [], transitions, {}, checkpoints, npeople=npeople)