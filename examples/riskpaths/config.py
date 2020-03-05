"""
RiskPaths
This will be a neworder implementation of the RiskPaths MODGEN model

See: 
https://www.statcan.gc.ca/eng/microsimulation/modgen/new/chap3/chap3
https://www.statcan.gc.ca/eng/microsimulation/modgen/new/chap4/chap4

  'RiskPaths is a simple, competing risk, case-based continuous time microsimulation model. Its
  main use is as a teaching tool, introducing microsimulation to social scientists and demonstrating
  how dynamic microsimulation models can be efficiently programmed using the language
  Modgen.
  Modgen is a generic microsimulation programming language developed and maintained at
  Statistics Canada.
  RiskPaths as well as the Modgen programming language and other related documents are
  available at www.statcan.gc.ca/microsimulation/modgen/modgen-eng.htm'

"""
import numpy as np
import neworder

population_size = 10000

neworder.timeline = neworder.Timeline(0, 100, [1])

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
 
# initialisation
neworder.initialisations = {
  "people": { "module": "riskpaths", "class_": "RiskPaths", "args": (population_size) }
}

neworder.transitions = {
  "a": "import sys; neworder.log(sys.argv)",
  "status": "people.pregnancy()"
}

# Finalisation 
neworder.checkpoints = {
  "1stats": "people.stats()",
  "2hist": "people.plot()"
}
