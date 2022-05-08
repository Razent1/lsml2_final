FROM python:3.9
ADD . /brocker
WORKDIR /brocker
RUN pip install -r requirements.txt
CMD celery -A brocker worker --loglevel=info