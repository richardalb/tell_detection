FROM nvidia/cuda:12.8.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
      git \
      python3-pip \
      python3-dev \
      python3-opencv \
      libglib2.0.0

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

WORKDIR /tell_detection
COPY ./logs/* /tell_detection/logs/

ENTRYPOINT ["python3"]
