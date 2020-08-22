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
import neworder
from riskpaths import RiskPaths

# serial mode
neworder.module_init(verbose=True)

population_size = 100000

timeline = neworder.Timeline(0, 100, [1])

riskpaths = RiskPaths(timeline, population_size)

neworder.run(riskpaths)
