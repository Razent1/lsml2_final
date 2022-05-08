import joblib
from time import sleep

from celery import Celery
from celery.result import AsyncResult

from flask import Flask, request, render_template, make_response

app = Flask(__name__)
model = joblib.load('gbc_model.mdl')
broker = Celery('tasks', backend='redis://redis', broker='redis://redis')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [[float(x) for x in request.form.values()]]
    except ValueError:
        return render_template('index.html', prediction_text='Incorrect input format')
    resp = broker.send_task('tasks.predict', int_features)
    pred = AsyncResult(resp.task_id, app=broker)
    while not pred.ready():
        sleep(0.5)
    if pred.result[0] == 0:
        output = "Setosa"
    elif pred.result[0] == 1:
        output = "Versicolor"
    else:
        output = "Virginica"

    return render_template('index.html', prediction_text='The species of Iris flower is {}'.format(output))


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
