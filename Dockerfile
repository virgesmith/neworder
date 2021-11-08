
FROM python:3.9

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends -y mpich libmpich-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# don't run as root
RUN groupadd -g 1729 no && useradd -r -u 1729 -g no no
RUN chown -R no:no /app
USER no

COPY ./examples /app/examples

ENV VENV=/app/venv
RUN python -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# install python deps
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r examples/requirements.txt

# just use the latest release, not dev
# RUN pip install -e . && python -m pytest && mpiexec -n 2 python -m pytest

ENV DISPLAY :0

# use docker run -it...
CMD bash
