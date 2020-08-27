"""
diagnostics
Prints diagnostic info and drops into an interactive shell
"""

import os
import subprocess
import neworder


class Diagnostics(neworder.Model):
  """
  Extends the neworder.Model class, adding a "step" that prints diagnostic information,
  and a ends by starting an interactive shell
  """
  def __init__(self, *args):
    super().__init__(neworder.Timeline.null(), neworder.MonteCarlo.deterministic_independent_seed)

  def step(self):
    neworder.log("neworder %s" % neworder.version())
    if neworder.embedded():
      # TODO this is correct only for a locally built embedded non-MPI version
      binary = "/src/bin/neworder"
    else:
      binary = neworder.__file__
    neworder.log(binary)
    all_libs = subprocess.getoutput("ldd %s" % binary).replace("\t", "").split("\n")
    neworder.log("Loaded libs:")
    for _, s in enumerate(all_libs): neworder.log("  %s" % s)
    neworder.log("PYTHONPATH=%s" % os.getenv("PYTHONPATH"))
    neworder.log("Globals: %s" % dir())

  def checkpoint(self):
    neworder.shell()

# log all
neworder.verbose()
# construct the "model"
model = Diagnostics()
# run it
neworder.run(model)

