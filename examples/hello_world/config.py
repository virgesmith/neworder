""" Hello World
The simplest functional neworder configuration
Serves as a skeleton for user projects
"""

# for shared arrays
import numpy as np
# Expose the enviroment to python
import neworder

# Timeline is compulsory, and must be float type - define a dummy timeline
neworder.timespan = np.array([0, 1], dtype=float)
# We only need one timestep
neworder.timestep = neworder.timespan[1]

# Checks to run during the simulation
loglevel = 1
do_checks = True
# checks only called once since there is only one timestep
checks = {
  # a do nothing-check purely for illustration - checks must evaluate to boolean 
  # Ony eval-able expressions allowed here.
  "eval": "True",
  #"exec": "a=True" # will fail, assigment is not eval-able
}

# Initialisation - construct an instance of Greet
#
# This creates an object called "greeter" within the neworder module, which is an instance of 
# the class Greet from the "greet" module and is initialised with no parameters.
# The pure python equivalent to the above is:
#   import greet
#   import neworder
#   neworder.greeter = greet.Greet()
initialisations = {
  "greeter": { "module": "greet", "class_": "Greet", "parameters": [] }
}

# The "transition" in this case fetches the current username from the os
# Note that the code is exec'd not eval'd: any return value is discarded
transitions = { 
  "who": "greeter.get_name()",
  "exec": "a=1" # won't fail. a is in neworder namespace
}

# Say hello when the empty simulation is done
#
# equivalent to 
# import neworder
# neworder.greeter()
# TODO control over order of execution...
checkpoints = {
  "exec": "b=a+1", # exec - shouldn't fail. a, b are in neworder namespace, and already initialised
  "print": "print(b)",
  "say_hello" : "greeter()",
}
