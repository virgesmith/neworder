"""
diagnostics.py

Prints diagnostic info
"""

import os
import sys
import subprocess
import numpy as np
import neworder


neworder.log("MODULE=" + neworder.name() + neworder.version())
neworder.log("PYTHON=" + neworder.python())

all_libs = subprocess.getoutput("ldd src/bin/neworder").replace("\t", "").split("\n")
neworder.log("Loaded neworder/boost/python libs:")
[neworder.log("  " + s) for _, s in enumerate(all_libs) if "neworder" in s or "python" in s or "boost" in s]

neworder.log("PYTHONPATH=" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "<undefined>")

# TODO more sophisitcated impl of the log level/checking 
loglevel = 1 
do_checks = False 
# note timeline must be represented by floats
neworder.timespan = np.array([0,1], dtype=float)
neworder.timestep = neworder.timespan[1]

initialisations = {}
transitions = {}

# finally, open an interactive shell
checkpoints = {
  "shell": "shell()"
}


