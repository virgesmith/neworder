""" Hello World
The simplest functional neworder configuration
Serves as a skeleton for user projects
"""

# Expose the enviroment to python
import neworder

# Timeline is compulsory - define a dummy timeline
neworder.timespan = neworder.DVector.fromlist([0, 1])
# We only need one timestep
neworder.timestep = neworder.timespan[1]

# No checks to run during the simulation
loglevel = 1
do_checks = True
# no per-timestep checks implemented since there is only one timestep
checks = {
  # a do nothing-check purely for illustration - checks must evaluate to boolean 
  "dummy": "True"
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
# Note that the code is exec'd not eval'd: any return value is ignored
transitions = { 
  "who": "greeter.get_name()"
}

# Say hello when the empty simulation is done
#
# equivalent to 
# import neworder
# neworder.greeter()
checkpoints = {
  "say_hello" : "greeter()"
}
