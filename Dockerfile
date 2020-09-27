
FROM python:3.8

WORKDIR /app

COPY . /app

RUN apt-get update -y && apt-get install -y --no-install-recommends -y mpich libmpich-dev

# install python deps
RUN python -m pip install -r examples-requirements.txt

# use setup.py to get pytest (not in requirements.txt)
RUN python setup.py install && python setup.py test && mpiexec -n 2 python setup.py pytest

ENV DISPLAY :0

# use docker run -it...
CMD ["bash"]
