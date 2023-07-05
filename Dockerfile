FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get -y upgrade && \
apt-get install -y ffmpeg && \
apt-get install -y git

COPY ./.devcontainer/requirements.txt .

RUN pip install -r ./requirements.txt