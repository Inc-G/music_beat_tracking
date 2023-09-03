"""
The main function is add_clicks, it adds the clicks to a track of music.
"""

import librosa 
import soundfile as sf

import numpy as np
import pandas as pd
import scipy

import models
import custom_metrics
import parameters as params 
import main_post_processing as post_processing

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf


def square(x):
    return x*x

vec_square = np.vectorize(square)


def predictions(song, model='', model_passed=True, model_loc='', len_frame=params.LEN_FRAME):
    audio, _ = librosa.load(song)
    
    #preprocessing
    S_original_audio = librosa.feature.melspectrogram(y=audio, sr=params.SAMPLING_RATE, n_mels=128)
    db_spectrogram = librosa.power_to_db(S_original_audio, ref=np.max)
    first_derivative = librosa.feature.delta(db_spectrogram)
    input_nn = np.array([np.concatenate((db_spectrogram, first_derivative)).transpose()])
    
    input_nn = vec_square(((input_nn) - params.MIN)/(params.MAX - params.MIN))
    
    if model_passed:
        result = model(input_nn[:,:len_frame,:])
    else:
        model = models.bidirectional_model()
        _ = tf.keras.models.load_model(model_loc)
        model(input_nn)
        model.set_weights(_.get_weights())
        result = model(input_nn[:,:len_frame,:]) 
    return result


def add_clicks(song, model='', model_passed=True, model_loc='', output_name='name.wav', constant_tempo=True, plot=False):
    """
    song: string. The location where there is the song.
    model_passed: bool. If True, then model should be the model used. If false, it will load a model from model_loc
    output_name: string. Location where the .wav file of the song + clicks will be saved.
    constant_tempo: bool. Whether the track has constant tempo or not. 
    plot: bool. If True, it will plot a bunch of graphs. Useful for debugging/understanding what main_post_processing does.

    Adds the clicks to the piece of music saved at song. It uses either the model passed, or the model saved at model_loc.
    """
    preds = predictions(song, model, model_passed, model_loc, len_frame=10*params.LEN_FRAME)
    
    beats = post_processing.frames_with_beat(preds, constant_tempo=constant_tempo, plot=plot)
    beats_f = custom_metrics.from_frames_to_times(beats)
    
    if plot:
        pd.DataFrame(np.array([preds.numpy()[0], beats]).transpose(), columns=['predictions', 'adjusted']).plot()
        plt.show()

    original_audio, sr = librosa.load(song)
    clicks_db = librosa.clicks(times=beats_f, sr=sr, length=len(original_audio))
    audio_and_clicks = clicks_db*1/2 + original_audio 
    sf.write(output_name, audio_and_clicks, sr)
    return preds, beats
