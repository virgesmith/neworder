""" Hello World
The simplest functional neworder configuration
"""

# Expose the enviroment to python
import neworder

# Timeline is compulsory - define a dummy timeline
neworder.timespan = neworder.DVector.fromlist([0, 1])
# We only need one timestep
neworder.timestep = neworder.timespan[1]

# No checks to run during the simulation
loglevel = 1
do_checks = False
# no per-timestep checks implemented since there is only one timestep
checks = { }

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

# There are no transitions to model
transitions = { }

# Say hello when the empty simulation is done
#
# This tells the framework to call the greeter object. The "say_hello" label is purely an internal identifier
# The pure python equivalent to the above is:
# neworder.greeter()
finalisations = {
  # __call__ is python-speak for call the object itself (like it was a function)
  "say_hello" : { "object": "greeter", "method": "__call__", "parameters": [] }
}
