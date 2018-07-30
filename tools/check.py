#!/usr/bin/env python3
# This does some basic checks on the model configuration

import os
import sys
import importlib.util
import pandas as pd
import neworder

def check(symbols, symbol_list):
  got_required = True
  for symbol in symbol_list:
    got_required = got_required and symbol in symbols
    print(symbol + "?", symbol in symbols)
  return got_required

  #return symbol in symbols

if len(sys.argv) != 2:
  print("usage: check.py config-file")
  exit(1)

have_required = True

modulename = sys.argv[1]
spec = importlib.util.spec_from_file_location(modulename, modulename)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

required_symbols = ["loglevel", "do_checks", "initialisations", "transitions", "finalisations"]
symbols = dir(module)

print("checking", modulename + "...")
check(symbols, required_symbols)

# symbols that are defined by the user, not by the embedded env
required_symbols = ["timespan", "timestep"]
symbols = dir(neworder)
print("checking neworder user definitions...")

check(symbols, required_symbols)

#print(symbols)
