from flask import Flask, render_template, request, redirect, url_for, jsonify

import os

import soundfile as sf
import librosa

import tensorflow as tf
import numpy as np
import models
import add_clicks

MODEL_NAME = 'model_epoch_300'

model = models.bidirectional_model()
model_save = models.bidirectional_model_for_save()
model(np.ones((1, 10, 256)))
model_save(np.ones((1, 10, 256)))

model_save.load_weights(MODEL_NAME+'/'+MODEL_NAME)
model.set_weights(model_save.get_weights())
print('loaded model!')

app = Flask(__name__)

# Specify the folder where uploaded files will be stored
UPLOAD_FOLDER = 'static/uploaded_songs'
FOLDER_WITH_CLICKS = 'with_clicks'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

constant_tempo = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return "File uploaded successfully"
    
    return "File upload failed"



@app.route('/get_beat', methods=['POST'])
def get_beat():
    
    song_name = request.get_json(force = True)
   
    loc = UPLOAD_FOLDER + '/' + song_name
    loc_clicks = 'static/'  + song_name[:-4] +'_clicks.wav'

    add_clicks.add_clicks(song=loc, model=model, model_passed=True, model_loc='',
        output_name=loc_clicks, constant_tempo=True, plot=False)
    
    return "beats added successfully"


if __name__ == '__main__':
    app.run(debug=True)
