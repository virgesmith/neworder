#!/bin/bash

. .env

version=$(grep "^version =" pyproject.toml | awk -F'"' '{print $2}')
echo $version

uv build --sdist


# TEST
uv publish -t $TEST_PYPI_API_TOKEN --publish-url https://test.pypi.org/legacy/ dist/neworder-$version.tar.gz

# PROD
# uv publish -t $PYPI_API_TOKEN dist/neworder-$version.tar.gz
