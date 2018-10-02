"""
diagnostics.py

Prints diagnostic info and drops into an interactive shell
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
neworder.log_level = 1 
neworder.do_checks = False 

# no timeline is defined

neworder.initialisations = {}
neworder.transitions = {}

# finally, open an interactive shell
neworder.checkpoints = {
  "shell": "shell()"
}


