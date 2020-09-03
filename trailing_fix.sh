#!/bin/bash

files=$(find docs/ -name "*.md")

for file in $files; do
  sed -i 's/[ \t]*$//' $file
done

files=$(find examples/ -name "*.md")

for file in $files; do
  sed -i 's/[ \t]*$//' $file
done