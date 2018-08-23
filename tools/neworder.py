
"""
Stub module for neworder

neworder is a module that exists only within the embedded python environment
This is a python stub for the module to enable neworder python code to load 
(but not run) in a standard python environment.
This can be useful for example to do basic sanity/syntax checking on model code.

You may need to ensure the path to this file is in your PYTHONPATH, e.g.
$ PYTHONPATH=tools python3 example/population.py
"""

# Stub DVector
class DVector:
  def fromlist(x):
    return list(x)

# Stub callback
class Callback:
  def __init__(self, string):
    pass

# Stub log
def name():
  return "stub"

def version():
  return "0"

def python():
  import sys
  return sys.version

def log(_):
  pass

# Default multiprocess settings
nprocs = 1
procid = 0
