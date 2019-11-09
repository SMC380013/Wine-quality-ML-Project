import os
import io
import numpy as np


import keras

from flask import Flask, request, redirect, render_template, url_for, jsonify

# from tensorflow.keras.models import load_model
# redwinequality_model = load_model("redwinequality_model_trained.h5")

from joblib import dump, load
redwinequality_model = load('redwinerandom.joblib') 

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print("-------------------HERE------------")
    final_features = [float(x) for x in request.form.values()]
    #final_features = np.reshape(final_features, 11)
    
    print('Final features', final_features)
#     prediction = redwinequality_model.predict_classes([final_features])
      prediction = redwinequality_model.predict([final_features])
    #prediction = [3]
    print("Prediction", prediction)

    output = prediction [0]

    return render_template('index.html', prediction_text='Wine Quality is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

