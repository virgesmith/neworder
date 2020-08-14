"""
Stub module for neworder

neworder is a module that exists only within the embedded python environment
This is a python stub for the module to enable neworder python code to load 
(but not run) in a standard python environment.
This can be useful for example to do basic sanity/syntax checking on model code.

You may need to ensure the path to this file is in your PYTHONPATH, e.g.
$ PYTHONPATH=tools python3 examples/hello_world/model.py
"""

# TODO decide if this file is useful and if so add all the neworder symbols

def name():
  return "stub"

def version():
  return "0.0.0"

def python():
  import sys
  return sys.version

def log(msg):
  print("[STUB] %s" % msg)
  
class Timeline:
  def __init__(self, _a, _b, _c):
    pass

  @staticmethod
  def null():
    return Timeline(0,0,0)


class Model:
  def __init__(self, _timeline):
    pass


def run(m):
  log("model %s not being run" % type(m))