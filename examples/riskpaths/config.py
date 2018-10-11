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

# TODO parameterise
mortality_rate = np.full(100, 0.002)
mortality_rate[-30:] = mortality_rate[-30:] * 50
fertility_rate = 0.014
p_u1f = np.full(50, 0.1) 
# if not wed by 50 will never happen (won't be fertile anyway)
p_u1f[-1] = 0.0
p_u1d = p_u1f # for now
p_u2f = p_u1f * 0.5 # 2nd union less likely
p_u2d = p_u2f 

population_size = 10000

# there is no timeline - this is the spacing the time-dep hazard rates
neworder.timestep = 1.0

# This timeline represents persons age
# range LIFE
#neworder.timeline = (0.0, 100.0, 1)

# running/debug options
neworder.log_level = 1
neworder.do_checks = False
# assumed to be methods of class_ returning True if checks pass
neworder.checks = {
  "check": "people.check()"
}
 
# initialisation
neworder.initialisations = {
  "people": { "module": "riskpaths", "class_": "RiskPaths", "parameters": [population_size, mortality_rate, p_u1f, p_u1d, p_u2f, p_u2d] }
}

neworder.transitions = {
  "status": "people.alive()"
}

# Finalisation 
neworder.checkpoints = {
  "hist": "people.plot()"
}
