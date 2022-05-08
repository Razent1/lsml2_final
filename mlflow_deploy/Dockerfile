FROM python:3.9

ADD . /mlfolow_deploy
WORKDIR /mlfolow_deploy

RUN pip install -r requirements.txt

ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

EXPOSE 5000

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000
