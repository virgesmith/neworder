"""
RiskPaths
This is a neworder implementation of the RiskPaths MODGEN model

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

from riskpaths import RiskPaths
from visualisation import plot

import neworder

# serial mode
# neworder.verbose()

population_size = 100000

# single step (continuous model)
riskpaths = RiskPaths(population_size)

neworder.run(riskpaths)

plot(riskpaths)
