""" 
Hello World
A very simple neworder model to introduce the basic concepts and workflow.
It subclasses neworder.Model adds implements a toy model which 
- gets the user's name, which is called during the "simulation"
- say hello, which is called at the end of the "simulation".
"""

# Expose the neworder enviroment to python
import neworder
import os

class HelloWorld(neworder.Model):
  """ 
  This model extends the builtin neworder.Model class by providing implementations of the following methods:
  - modify (optional)
  - transition
  - check (optional)
  - checkpoint
  The neworder runtime will automatically execute the model, looping over the timeline and calling the methods above  
  """

  def __init__(self):
    """
    In this example we don't have an explicit timeline, 
    We create a null timeline, corresponding to a single instantaneous transition,
    and initialised the base class with this 
    """
    super().__init__(neworder.Timeline.null())
    self.name = "unknown"

  # def modify(self, rank):
  #   """ 
  #   For parallel runs only, per-process state modifications can be made before the model, allowing 
  #   for e.g. sensitivity analysis or splitting datasets across parallel model runs
  #   This method is optional.
  #   Arguments: self, rank (MPI process number)
  #   Returns: NoneType
  #   """
  #   pass

  def transition(self):
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
    return self.name != "unknown"

  def checkpoint(self):
    """
    Checkpoints are run at each checkpoint (the final timestep is always a checkpoint) 
    This method must be implemented.
    Arguments: self
    Returns: NoneType
    """
    neworder.log("Hello %s" % self.name)
  
# construct the model
neworder.model = HelloWorld()


