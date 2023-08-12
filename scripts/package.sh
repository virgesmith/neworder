#!/bin/bash

. .env

version=$(grep -e "^version" pyproject.toml | cut  -d' ' -d'=' -d'"' - -f2)

# package as source dist
python -m build . --sdist

# upload
twine upload -u __token__ -p $TEST_PYPI_API_TOKEN  --repository-url https://test.pypi.org/legacy/ dist/neworder-$version.tar.gz
# twine upload -u __token__ -p $PYPI_API_TOKEN --repository-url https://upload.pypi.org/legacy/ dist/neworder-$version.tar.gz

# NB on testPyPI, deps need to been installed from main repo. Use this:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple neworder