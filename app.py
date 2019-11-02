import os
import io
import numpy as np


import keras

from flask import Flask, request, redirect, render_template, url_for, jsonify

from tensorflow.keras.models import load_model
redwinequality_model = load_model("redwinequality_model_trained.h5")

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
    final_features = [int(x) for x in request.form.values()]
    #final_features = np.reshape(final_features, 11)
    
    print('Final features', final_features)
    prediction = redwinequality_model.predict([final_features])
    #prediction = [3]
    print("Prediction", prediction)

    output = prediction[0]

    return render_template('index.html', prediction_text='Wine Quality is {}'.format(output))


# app.config['UPLOAD_FOLDER'] = 'Uploads'

# model = None
# graph = None


# def load_model():
#     global model
#     global graph
#     model = Xception(weights="imagenet")
#     graph = K.get_session().graph


# load_model()


# def prepare_image(img):
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     # return the processed image
#     return img


# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     data = {"success": False}
#     if request.method == 'POST':
#         if request.files.get('file'):
#             # read the file
#             file = request.files['file']

#             # read the filename
#             filename = file.filename

#             # create a path to the uploads folder
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#             file.save(filepath)

#             # Load the saved image using Keras and resize it to the Xception
#             # format of 299x299 pixels
#             image_size = (299, 299)
#             im = keras.preprocessing.image.load_img(filepath,
#                                                     target_size=image_size,
#                                                     grayscale=False)

#             # preprocess the image and prepare it for classification
#             image = prepare_image(im)

#             global graph
#             with graph.as_default():
#                 preds = model.predict(image)
#                 results = decode_predictions(preds)
#                 # print the results
#                 print(results)

#                 data["predictions"] = []

#                 # loop over the results and add them to the list of
#                 # returned predictions
#                 for (imagenetID, label, prob) in results[0]:
#                     r = {"label": label, "probability": float(prob)}
#                     data["predictions"].append(r)

#                 # indicate that the request was a success
#                 data["success"] = True

#         return jsonify(data)

#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form method=post enctype=multipart/form-data>
#       <p><input type=file name=file>
#          <input type=submit value=Upload>
#     </form>
#     '''


if __name__ == "__main__":
    app.run(debug=True)
