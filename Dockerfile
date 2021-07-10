
FROM python:3.8

WORKDIR /app

COPY . /app

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends -y mpich libmpich-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install python deps
RUN pip install -U pip pytest
RUN python -m pip install -r examples/requirements.txt

# use setup.py to get pytest (not in requirements.txt)
RUN pip install -e . && python -m pytest && mpiexec -n 2 python -m pytest

ENV DISPLAY :0

# use docker run -it...
CMD ["bash"]
