#!/bin/bash

CFLAGS=--coverage python -m pip install -e .
pytest

bash <(curl -s https://codecov.io/bash)

