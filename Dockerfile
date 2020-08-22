
FROM python:3.8

WORKDIR /app

COPY . /app

#RUN apt-get update -y && apt-get install -y --no-install-recommends -y libspatialindex-dev

# install python deps
RUN python -m pip install -r requirements.txt

RUN python setup.py install && python setup.py test
