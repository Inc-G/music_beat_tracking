import os
import shutil
import pickle

import jams
import librosa
import soundfile as sf

import parameters as params

import numpy as np


MEL_SAMPLING_RATE = params.MEL_SAMPLING_RATE
SAMPLING_RATE = params.SAMPLING_RATE
MAX_LEN_SONG = 1287

ANNOTATIONS_LOC = 'GTZAN_annotations/jams/' #This should be the folder where there are the annotations
SONGS_LOC = 'GTZAN/genres_original/' #This should be the folder with the songs from GTZAN dataset

train_songs = set(np.random.choice(list(range(100)), 75, False))


print('Creating the folders train and test, and moving 1/4 songs to test and 3/4 to train')
os.makedirs(SONGS_LOC+'train')
os.makedirs(SONGS_LOC+'test')

## Cannot open 'jazz.00054.wav'

for genre in os.listdir(SONGS_LOC):
    if genre[0]!= '.' and genre != 'train' and genre != 'test':
        num_files = len(os.listdir(SONGS_LOC+genre))
        training_set = set(np.random.choice(list(range(num_files)), 3*num_files//4, False))
        for idx, song in enumerate(os.listdir(SONGS_LOC+genre)):
            if idx in training_set and song != 'jazz.00054.wav':
                shutil.copy(SONGS_LOC + genre + '/' + song, SONGS_LOC + 'train')
            else:
                if song != 'jazz.00054.wav':
                    shutil.copy(SONGS_LOC + genre + '/' + song, SONGS_LOC + 'test')
                

print('Getting the annotations for train and test')
        
annotations_dic = {}

for song in os.listdir(ANNOTATIONS_LOC):
    annotations_dic[song[:-5]] = []
    jam = jams.load(ANNOTATIONS_LOC + song)
    for annotation in jam.annotations:
        if annotation.namespace == "beat":
            for obs in annotation.data:
                annotations_dic[song[:-5]].append(obs.time)
            break


annotations_train = {}
annotations_test = {}

for song in os.listdir(SONGS_LOC + 'train'):
    new = np.zeros(MAX_LEN_SONG)
    for _ in np.round(np.array(annotations_dic[song])*MEL_SAMPLING_RATE):
        if _ < MAX_LEN_SONG:
            new[int(_)] = 1
    annotations_train[song] = new

for song in os.listdir(SONGS_LOC+'test'):
    new = np.zeros(MAX_LEN_SONG)
    for _ in np.round(np.array(annotations_dic[song])*MEL_SAMPLING_RATE):
        if _ < MAX_LEN_SONG:
            new[int(_)] = 1
    annotations_test[song] = new


print('Preprocessing the songs in train and test folder')
mells_train = {}
mells_test = {}


for song in os.listdir(SONGS_LOC + 'test'):
    if song[0] !='.':
        audio, _ = librosa.load(SONGS_LOC + 'test/' + song, sr=SAMPLING_RATE)
        S_original_audio = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE, n_mels=128)
        db_spectrogram = librosa.power_to_db(S_original_audio, ref=np.max)
        first_derivative = librosa.feature.delta(db_spectrogram)
    
        mells_test[song] = np.concatenate((db_spectrogram, first_derivative))[:,:MAX_LEN_SONG] 
        


for song in os.listdir(SONGS_LOC + 'train'):
    if song[0] !='.':
        audio, _ = librosa.load(SONGS_LOC + 'train/' + song, sr=SAMPLING_RATE)
        S_original_audio = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE, n_mels=128)
        db_spectrogram = librosa.power_to_db(S_original_audio, ref=np.max)
        first_derivative = librosa.feature.delta(db_spectrogram)
    
        mells_train[song] = np.concatenate((db_spectrogram, first_derivative))[:,:MAX_LEN_SONG] 


## Preprocessing functions
MIN = -81
MAX = 13

def square(x):
    return x*x

vec_square = np.vectorize(square)


print('Saving the preprocessed songs for both train and test in' + SONGS_LOC + 'train/ and '+ SONGS_LOC + 'test/')

training_input = np.stack([vec_square((_ - MIN)/(MAX - MIN)) for _ in mells_train.values()])
np.save(SONGS_LOC + 'train/' +'transformed_inputs.npy', training_input)
del training_input

test_input = np.stack([vec_square((_ - MIN)/(MAX - MIN)) for _ in mells_test.values()])
np.save(SONGS_LOC + 'test/' +'transformed_inputs_test.npy', test_input)
del test_input

print('Saving the targets for both train and test')

training_target = np.stack([annotations_train[song] for song in mells_train.keys()])
np.save(SONGS_LOC + 'train/' +'training_target.npy', training_target)

test_target = np.stack([annotations_test[song] for song in mells_test.keys()])
np.save(SONGS_LOC + 'test/' +'test_target.npy', test_target)         
del training_target, test_target
