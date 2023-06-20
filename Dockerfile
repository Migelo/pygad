FROM python:3.11-slim-buster

LABEL maintainer="Miha Cernetic <cernetic@mpa-garching.mpg.de>"

# The enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1

WORKDIR "/data"

RUN set -ex  && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgsl-dev g++ gcc build-essential wget git && \
    /usr/local/bin/python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY . /data/pygad
RUN set -ex \
    useradd -ms /bin/bash runner && \
    chown -R runner /data

USER runner
ENV PATH="/home/runner/.local/bin:${PATH}"

WORKDIR "/data/pygad"

RUN set -ex \
    pwd && \
    ls -la && \
    pip install -e . && \
    pip install --user jupyter && \
    pip cache purge && \
    echo -e "\n\nSUCCESS: done installing pygad and jupyter."

# Fix path to the test snapshots
RUN wget https://bitbucket.org/broett/pygad/downloads/QuickStart-v0.9.10-2021-06-24.ipynb && \
    sed -i "s#'./snaps/#pg.__path__[0]+'/snaps/#g" *.ipynb

HEALTHCHECK --interval=1m --timeout=3s \
CMD curl --fail http://localhost:1337 || exit 1

# ENTRYPOINT ["/data/entrypoint.sh"]
CMD ["jupyter", "notebook", "--ip=*", "--port=1337", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password="]

# Expose the port of the Docker container
EXPOSE 1337
