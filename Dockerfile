FROM python:3.9 

RUN apt-get update && apt-get upgrade && \
apt-get install -y ffmpeg

COPY ./.devcontainer/requirements.txt .

RUN pip install -r ./requirements.txt