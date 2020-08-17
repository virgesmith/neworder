"""
diagnostics
Prints diagnostic info and drops into an interactive shell
"""

import os
import subprocess
import neworder

neworder.module_init(0,1,True, True)

class Diagnostics(neworder.Model):
  """
  Extends the neworder.Model class, adding a "transition" that prints diagnostic information,
  and a ends by starting an insteractive shell
  """
  def __init__(self, *args):
    super().__init__(neworder.Timeline.null())


  def transition(self):
    neworder.log("MODULE= %s %s" % (neworder.name(), neworder.version()))
    binary = "target/debug/neworder" if (neworder.name() == "neworder.rs") else "src/bin/neworder"
    all_libs = subprocess.getoutput("ldd %s" % binary).replace("\t", "").split("\n")
    neworder.log("Loaded libs:")
    for _, s in enumerate(all_libs): neworder.log("  %s" % s) 
    neworder.log("PYTHONPATH=%s" % os.getenv("PYTHONPATH"))
    neworder.log("Globals: %s" % dir())

  def checkpoint(self):
    neworder.shell()


# construct the model
model = Diagnostics()

neworder.run(model)


