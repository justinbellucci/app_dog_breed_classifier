import os
import io
from flask import Flask, render_template, request, redirect

import backend.detectors as detectors
from backend.predict import predict_breed
# from backend.image_helper import show_img, process_image
from backend.model_functions import load_class_names

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url) # redirect to same page
        file = request.files['file'] # get file from request
        if not file:
            return # add error message
    
        # process image
        img_bytes = file.read()
        # run image throuh model
        is_dog = detectors.dog_detector(io.BytesIO(img_bytes))
        probs, classes = predict_breed(io.BytesIO(img_bytes))
        class_names = load_class_names()

        # add model output to JSON files
        dog_detected = {"is_dog": is_dog}
        dog_breeds = [{"breed": class_names[classes[0]], "prob": probs[0]},
                      {"breed": class_names[classes[1]], "prob": probs[1]},
                      {"breed": class_names[classes[2]], "prob": probs[2]},
                      {"breed": class_names[classes[3]], "prob": probs[3]},
                      {"breed": class_names[classes[4]], "prob": probs[4]}]

        return render_template('results.html', title='Results', dog_detected=dog_detected, dog_breeds=dog_breeds)
    return render_template('index.html', title='Index')

if __name__ == "__main__":
    app.run(debug=True)