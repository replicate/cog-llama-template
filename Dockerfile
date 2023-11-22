FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 curl
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.1.1/pget" && chmod +x /usr/local/bin/pget
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN pip install transformers==4.33.2 torch --pre -f https://mlc.ai/wheels mlc-chat-nightly-cu121 mlc-ai-nightly-cu121
WORKDIR /app
COPY ./repro_utils.py ./reproduce.py /app/
ENTRYPOINT ["/usr/bin/python3.11", "reproduce.py"]
