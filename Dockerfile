# syntax=docker/dockerfile:1
FROM gcr.io/datamechanics/spark@sha256:bbf0f840b485b512d0da531ab69f46ba7f443e10821d9dda5c4ca47b4b221753

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/application

COPY requirements.txt .
RUN pip3 install --upgrade pip --user
RUN pip3 install -r requirements.txt --user

COPY . .