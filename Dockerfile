FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7
RUN apk --update add bash nano
FROM tensorflow/tensorflow
COPY main.py main.py
COPY get_prediction.py get_prediction.py
COPY cls cls
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
RUN pip install --upgrade pip==20.3.3
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt