FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip build-essential

COPY . /app
WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
