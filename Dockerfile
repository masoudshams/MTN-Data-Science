FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install opencv-python-headless

RUN pip install -r requirements.txt

COPY Shams-Platinum.py .

CMD python Shams-Platinum.py