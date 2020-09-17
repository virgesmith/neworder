"""
Hello World
A very simple neworder model to introduce the basic concepts and workflow.
It subclasses neworder.Model adds implements a toy model which
- gets the user's name, which is called during the "simulation"
- say hello, which is called at the end of the "simulation".
"""

#!class!
import pandas as pd
import neworder

class HelloWorld(neworder.Model):
  """
  This model extends the builtin neworder.Model class by providing 
  implementations of the following methods:
  - modify (optional)
  - step
  - check (optional)
  - checkpoint
  The neworder.run() function will execute the model, looping over 
  the timeline and calling the methods above
  """
#!class!

#!constructor!
  def __init__(self, n, p):
    """
    We create a null timeline, corresponding to a single instantaneous 
    transition, and initialise the base class with this plus a 
    randomly-seeded Monte-Carlo engine

    NB it is *essential* to initialise the base class.
    Failure to do so will result in UNDEFINED BEHAVIOUR
    """
    super().__init__(neworder.Timeline.null(), 
                     neworder.MonteCarlo.nondeterministic_stream)

    # create a silent population of size n
    self.population = pd.DataFrame(index=neworder.df.unique_index(n), 
                                   data={"talkative": False}) 
    self.population.index.name = "id"

    # set the transition probability
    self.p_talk = p
#!constructor!

  # def modify(self, rank):
  #   """
  #   For parallel runs only, per-process state modifications can be 
  #   made before the model runs, allowing for e.g. sensitivity analysis 
  #   or splitting datasets across parallel model runs
  #   This method is optional.
  #   Arguments: self, rank (MPI process number)
  #   Returns: NoneType
  #   """
  #   pass

  def __str__(self):
    """
    Returns a more readable name for verbose logging output, would 
    otherwise be something like
    "<__main__.HelloWorld object at 0x7fe82792da90>"
    """
    return self.__class__.__name__

  #!step!
  def step(self):
    """
    Transitions to run at each timestep.
    This method must be implemented.
    Arguments: self
    Returns: NoneType
    """
    # randomly make some people talkative
    self.population.talkative = self.mc().hazard(self.p_talk, len(self.population)).astype(bool)
  # !step!

  # !checkpoint!
  def checkpoint(self):
    """
    Checkpoints are run at each checkpoint 
    (NB the final timestep is always a checkpoint)
    This method must be implemented.
    Arguments: self
    Returns: NoneType
    """
    for i, r in self.population.iterrows():
      if r.talkative: neworder.log("Hello from %d" % i)
  # !checkpoint!

  # def check(self):
  #   """
  #   Custom checks can be made after every timestep during the simulation.
  #   This method is optional
  #   Arguments: self
  #   Returns: bool
  #   """
  #   return True

#!script!
# uncomment for verbose output
neworder.verbose()
# uncomment to disable checks entirely
# neworder.checked(False)

# construct the model with the population size and the chances to talking
hello_world = HelloWorld(10, 0.5)

# run the model and check it worked
ok = neworder.run(hello_world)
if not ok: neworder.log("model failed!")
#!script!
