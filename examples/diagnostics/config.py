"""
diagnostics.py

Prints diagnostic info and drops into an interactive shell
"""

import os
import sys
import subprocess
#import numpy as np
import neworder

neworder.log("MODULE= %s %s" % (neworder.name(), neworder.version()))

binary = "target/debug/neworder" if (neworder.name() == "neworder.rs") else "src/bin/neworder"

all_libs = subprocess.getoutput("ldd %s" % binary).replace("\t", "").split("\n")
neworder.log("Loaded libs:")
[neworder.log("  " + s) for _, s in enumerate(all_libs)]

neworder.log("PYTHONPATH=" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "<undefined>")

# TODO more sophisitcated impl of the log level/checking 
neworder.log_level = 1 
neworder.do_checks = False 

# null timeline
neworder.timeline = neworder.Timeline.null()

neworder.log(str(neworder.timeline))

neworder.initialisations = {}
neworder.transitions = {}

# finally, open an interactive shell
neworder.checkpoints = {
  # TODO unify C++/rust versions
  "shell": "neworder.shell()"
  #"shell": "shell()"
}


