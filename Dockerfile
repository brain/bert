FROM python:3.6.2

WORKDIR /brain/src

ADD requirements.txt /brain/src
RUN pip install -r requirements.txt

ADD . /brain/src

EXPOSE 80

CMD ["python"]
