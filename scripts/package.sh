#!/bin/bash

version=$(cat VERSION) 

# package as source dist
python setup.py sdist 
# upload
#twine upload --repository-url https://test.pypi.org/legacy/ dist/neworder-$version.tar.gz
twine upload --repository-url https://upload.pypi.org/legacy/ dist/neworder-$version*

