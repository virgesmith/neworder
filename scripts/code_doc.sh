#!/bin/bash

VERSION=$(cat neworder/__init__.py |grep __version__|cut  -d' ' -d'=' -d'"' - -f2)

sed -e "s/VERSION/${VERSION}/g" docs/examples/src.md_template > docs/examples/src.md

# generate api doc -> apidoc.md
python scripts/docstr2md.py

# zip example code into docs folder
find ./examples -type d -name __pycache__ -o -name output > excluded
tar zcfv docs/examples/neworder-${VERSION}-examples-src.tgz -X excluded ./examples
rm excluded
zip -r docs/examples/neworder-${VERSION}-examples-src.zip examples -x "*/__pycache__/*" -x "*/output/*"
