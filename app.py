from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
cwd = os.getcwd()

model_weight = cwd + '/models/model.h5'
model = load_model(model_weight)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_prediction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    image_data = np.expand_dims(img_array, axis=0)
    image_data = preprocess_input(image_data)
    pred = model.predict(image_data)

    return pred


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        img_file = request.files['file']

        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(img_file.filename))
        img_file.save(file_path)

        # Prediction

        result = model_prediction(file_path, model)
        time.sleep(1)
        if result[0][0] == 1:
            return 'Negative (Non-Tumorous)'
        else:
            return 'Positive (Tumorous)'
    return None


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
