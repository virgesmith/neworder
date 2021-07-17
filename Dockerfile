
FROM python:3.9

WORKDIR /app

COPY ./examples /app/examples

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends -y mpich libmpich-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install python deps
RUN pip install -U pip
RUN pip install -r examples/requirements.txt

# just use the latest release
# RUN pip install -e . && python -m pytest && mpiexec -n 2 python -m pytest

ENV DISPLAY :0

# use docker run -it...
CMD ["bash"]
