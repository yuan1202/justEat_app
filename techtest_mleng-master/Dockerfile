FROM python:3.9-buster

LABEL maintainer="Just Eat AI & Algos <aiandalgorithms@just-eat.com>"

# Install python packages
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN which python
RUN rm -rf /root/.cache && rm requirements.txt

ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL en_GB.UTF-8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image

COPY wsgi.py /opt/program/
COPY src /opt/program/
COPY data /opt/data/
WORKDIR /opt/program

ENTRYPOINT ["/usr/local/bin/python", "wsgi.py"]