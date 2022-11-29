FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV LISTEN_PORT=5000
EXPOSE 5000

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip build-essential

WORKDIR /app
COPY . /app

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
