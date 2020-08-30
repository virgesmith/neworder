#!/bin/bash

files=$(find examples/ -name "*.py")

for file in $files; do
  sed -i 's/[ \t]*$//' $file
done