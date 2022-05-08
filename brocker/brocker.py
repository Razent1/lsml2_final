from dataclasses import dataclass

import joblib
import numpy as np
from celery import Celery

celery_app = Celery('tasks', backend='redis://redis', broker='redis://redis')
model = joblib.load('gbc_model.mdl')


@celery_app.task(name='tasks.predict')
def predict(features):
    result = model.predict(np.array([features])).tolist()
    return result
