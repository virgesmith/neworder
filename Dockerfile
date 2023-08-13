
FROM python:3.11

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends -y mpich libmpich-dev tk-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


# don't run as root
RUN groupadd -g 1729 no && useradd -m -u 1729 -g no no
USER no
WORKDIR /home/no

COPY ./examples /home/no/examples

ENV VENV=/home/no/venv
RUN python -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# install python deps
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir neworder[parallel,geospatial]

ENV DISPLAY :0

# use docker run -it...
CMD bash
