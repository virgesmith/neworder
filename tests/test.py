""" Test utils """

import inspect
import neworder

any_failed = False

def check(expr):
  if not expr:
    trace = inspect.stack()[1]
    neworder.log("FAIL %s at %s:%d" % (trace.code_context[0].strip("\n"), trace.filename, trace.lineno)) #["code_context"])
    any_failed = True
