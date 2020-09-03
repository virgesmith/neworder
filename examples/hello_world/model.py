"""
Hello World
A very simple neworder model to introduce the basic concepts and workflow.
It subclasses neworder.Model adds implements a toy model which
- gets the user's name, which is called during the "simulation"
- say hello, which is called at the end of the "simulation".
"""

import os
import neworder

# seeding strategy example as a function
def _seeder(_):
  return 19937

# seeding strategy as an anonymous function
_seeder2 = lambda r: 19937 + r

class HelloWorld(neworder.Model):
  """
  This model extends the builtin neworder.Model class by providing implementations of the following methods:
  - modify (optional)
  - step
  - check (optional)
  - checkpoint
  The neworder runtime will automatically execute the model, looping over the timeline and calling the methods above
  """

  def __init__(self):
    """
    In this example we don't have an explicit timeline,
    We create a null timeline, corresponding to a single instantaneous transition,
    and initialised the base class with this

    NB it is *essential* to initialise the base class.
    Failing to do so will result in UNDEFINED BEHAVIOUR
    """
    super().__init__(neworder.Timeline.null(), neworder.MonteCarlo.deterministic_independent_stream)
    self.name = None

  # def modify(self, rank):
  #   """
  #   For parallel runs only, per-process state modifications can be made before the model runs, allowing
  #   for e.g. sensitivity analysis or splitting datasets across parallel model runs
  #   This method is optional.
  #   Arguments: self, rank (MPI process number)
  #   Returns: NoneType
  #   """
  #   pass

  def __str__(self):
    """
    Returns a more readable name for verbose logging output, would otherwise be something like
    "<__main__.HelloWorld object at 0x7fe82792da90>"
    """
    return self.__class__.__name__

  def step(self):
    """
    Transitions to run at each timestep.
    This method must be implemented.
    Arguments: self
    Returns: NoneType
    """
    self.name = os.getlogin()

  def check(self):
    """
    Checks can be made after every timestep during the simulation.
    This method is optional
    Arguments: self
    Returns: bool
    """
    return self.name is not None

  def checkpoint(self):
    """
    Checkpoints are run at each checkpoint (the final timestep is always a checkpoint)
    This method must be implemented.
    Arguments: self
    Returns: NoneType
    """
    neworder.log("Hello %s" % self.name)

# uncomment for verbose output
#neworder.verbose()

# construct the model
hello_world = HelloWorld()

# run the model and check it worked
ok = neworder.run(hello_world)
if not ok: neworder.log("model failed!")