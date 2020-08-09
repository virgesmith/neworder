"""
diagnostics.py

Prints diagnostic info and drops into an interactive shell
"""

import os
import sys
import subprocess
#import numpy as np
import neworder

# null timeline
timeline = neworder.Timeline.null()

# nothing to transition
transitions = {}

# at end of simulation, open an interactive shell
checkpoints = {
  # TODO unify C++/rust versions
  "shell": "neworder.shell()"
  #"shell": "shell()"
}

class Diagnostics(neworder.Model):
  """
  Extends the neworder.Model class, adding a method called "info"
  NB this method is NOT visible to the neworder runtime, so it must be explicitly called from python
  """
  def __init__(self, *args):
    super().__init__(*args)
  
  def info(self):
    neworder.log("MODULE= %s %s" % (neworder.name(), neworder.version()))

    binary = "target/debug/neworder" if (neworder.name() == "neworder.rs") else "src/bin/neworder"

    all_libs = subprocess.getoutput("ldd %s" % binary).replace("\t", "").split("\n")
    neworder.log("Loaded libs:")
    [neworder.log("  %s" % s) for _, s in enumerate(all_libs)]
    neworder.log("PYTHONPATH=" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "<undefined>")


# construct the model
neworder.model = Diagnostics(
  timeline,
  [], 
  transitions,
  {},
  checkpoints
  )

# explicitly call info method
neworder.model.info()



