#!/usr/bin/env python3
# This does some basic checks on the model configuration

import os
import sys
import importlib.util
import pandas as pd

if len(sys.argv) != 2:
  print("usage: check.py config-file")
  exit(1)

modulename = sys.argv[1]
spec = importlib.util.spec_from_file_location(modulename, modulename)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

required_symbols = [ "module", "class_", "parameters", "transitions", "output"]
symbols = dir(module)

print("checking", modulename + "...")

for symbol in required_symbols:
  print(symbol + "?", symbol in symbols)

#print(symbols)
