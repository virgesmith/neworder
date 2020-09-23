#!/bin/bash

# generate api doc -> apidoc.md
python scripts/docstr2md.py

# zip example code into docs folder
find ./examples -type d -name __pycache__ -o -name output > excluded
tar zcfv docs/examples/neworder-examples-src.tgz -X excluded ./examples
rm excluded
zip -r docs/examples/neworder-examples-src.zip examples -x "*/__pycache__/*" -x "*/output/*"
