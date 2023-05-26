FROM --platform=linux/amd64 python:3.10 AS base
LABEL maintainer="Marco Rossi <rssmrc.11@gmail.com>"
ENV PYTHONPATH "${PYTHONPATH}:/home/ai/"
WORKDIR /home/ai
COPY requirements.txt .
RUN mkdir data; pip install --upgrade pip && pip install -r requirements.txt
COPY ./src ./src
CMD python -m src.main
