""" 
Hello World
A very simple neworder model to introduce the basic concepts and workflow.
It constructs a greeter object, which 
- gets the user's name
- says hello 
"""

# Expose the neworder enviroment to python
import neworder

# neworder.log_level = 1 # this doesnt do anything at the moment

# In this example we don't have a discrete timeline, 
# but we needs to explicitly specify that this is the case
timeline = neworder.Timeline.null()

# Checks, if specified, are run after every timestep during the simulation
checks = {
  # a do nothing-check purely for illustration - checks must evaluate to boolean 
  # Ony eval-able expressions allowed here.
  "eval": "True",
}

# Initialisation - construct an instance of the Greet class
from greet import Greet # allowed because PYTHONPATH is set explicitly to the directory
initialisations = { "greeter": Greet("Namaste", "Bonjour", "Hola", "Annyeonghaseyo") }


# The "transition" in this case fetches the current username from the os
transitions = { 
  "who": "greeter.set_name()"
}

# Say hello when the empty simulation is done
#
# equivalent to 
# import neworder
# neworder.greeter()
# TODO control over order of execution...list of tuples?
checkpoints = {
  "say_hello" : "greeter()",
}

# this model could extend the builtin one
class HelloWorld(neworder.Model):
  def __init__(self, *args):
    super().__init__(*args)

# construct the model
neworder.model = HelloWorld(
  timeline,
  [], # no modifiers
  initialisations,
  transitions,
  checks,
  checkpoints
  )

# this will happen automatically
#model.run()

