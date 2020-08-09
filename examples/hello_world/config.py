""" 
Hello World
A very simple neworder model to introduce the basic concepts and workflow.
It subclasses neworder.Model adding methods to 
- gets the user's name, which is called during the "simulation"
- say hello, which is called at the end of the "simulation".
"""

# Expose the neworder enviroment to python
import neworder

# In this example we don't have a discrete timeline, 
# but we need to explicitly specify that this is the case
# a null timeline corresponds to a single instantaneous transition
timeline = neworder.Timeline.null()

# Checks, if specified, are run after every timestep during the simulation
checks = {
  # a do nothing-check purely for illustration - checks must evaluate to boolean 
  # check-expressions must evaluate to a boolean.
  "eval": "True",
}

# The "transition" in this case fetches the current username from the os
transitions = { 
  "who": "neworder.model.set_name()"
}

neworder.log(dir())

# Say hello when the (empty) simulation is done
checkpoints = {
  "say_hello" : "neworder.model()"
}

import os
# this model extends the builtin one by adding the set_name and __call__ methods
class HelloWorld(neworder.Model):
  def __init__(self, *args):
    super().__init__(*args)
    self.name = "unknown"

  # Gets username
  def set_name(self):
    self.name = os.getlogin()

  def __call__(self):
    neworder.log("Hello %s" % self.name)
  
# construct the model
neworder.model = HelloWorld(
  timeline,
  [], # no modifiers
  transitions,
  checks,
  checkpoints
  )

# this will happen automatically
#model.run()

