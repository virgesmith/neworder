#!/bin/bash

CFLAGS="--coverage" ./setup.py install
pytest

bash <(curl -s https://codecov.io/bash)

