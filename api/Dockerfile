FROM python:3.9
ADD . /api
WORKDIR /api
RUN pip install -r requirements.txt
EXPOSE 9091
CMD python /api/api.py