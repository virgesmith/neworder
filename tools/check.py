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
    print("  " + symbol + "?", symbol in symbols)
  return got_required

def check_opt(symbols, symbol_list):
  #got_required = True
  for symbol in symbol_list:
    if hasattr(neworder, symbol):
      print("  " + symbol + ":", getattr(neworder, symbol))
    else:
      print("  " + symbol + ": [undefined]")

if len(sys.argv) != 2:
  print("usage: check.py config-file")
  exit(1)

have_required = True

modulename = sys.argv[1]
spec = importlib.util.spec_from_file_location(modulename, modulename)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

symbols = dir(neworder)

required_symbols = ["timespan", "timestep", "initialisations", "transitions", "checkpoints"]

print("checking required", modulename + "...")
have_required = have_required and check(symbols, required_symbols)

optional_symbols = ["rank", "size", "log_level", "do_checks", "checks"] 

print("checking optional", modulename + "...")
check_opt(symbols, optional_symbols)

print("CHECK OK" if have_required else "MISSING DEFINITION(s)")
