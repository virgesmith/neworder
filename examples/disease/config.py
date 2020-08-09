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

initialisations = {
  "graphics": Graphics() 
}

transitions = {
  "model": "neworder.model.step()"
}

checkpoints = {
  "finalise": "neworder.model.finalise()",
  "graphics": "graphics.plot(neworder.model)"
}

neworder.model = DiseaseModel(timeline, [], initialisations, transitions, {}, checkpoints, npeople=npeople)