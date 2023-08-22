import os
import shutil
import pickle

import jams
import librosa
import soundfile as sf

import numpy as np


MEL_SAMPLING_RATE = params.MEL_SAMPLING_RATE
SAMPLING_RATE = params.SAMPLING_RATE

ANNOTATIONS_LOC = 'BallroomAnnotations-master/'
SONGS_LOC = 'BallroomData'

GENRES = [SONGS_LOC+'/'+_ for _ in os.listdir(SONGS_LOC) if _ != '.DS_Store' and _!='allBallroomFiles' and _!='nada']

names_songs_in_annotations = set(os.listdir(ANNOTATIONS_LOC))
names_songs_in_annotations.remove('README.md')
names_songs = []
for _ in names_songs_in_annotations:
    names_songs.append(_[:-6])


## Creates four dictionaries:
#temps: (name of the song) --> times in seconds which have a beat
#frames: (name of the song) --> frames which have a beat
#time_first_beat: (name of the song) --> times in seconds which has the first beat. 
#frame_first_beat: (name of the song) --> frame which has the first beat. 

temps = {}
frames = {}
time_first_beat = {}
frame_first_beat = {}


for name in names_songs:
    file = open(ANNOTATIONS_LOC + name + '.beats')
    beats_in_seconds = file.read()
    temps[name] = []    
    first_beat_recorded = False
    for time in beats_in_seconds.split('\n'):        
        if len(time) > 0:
            temps[name].append(float(time[:-2]))
            if not first_beat_recorded:
                time_first_beat[name] = float(time[:-2])
                first_beat_recorded = True
    frames[name] = librosa.time_to_samples(temps[name], sr=SAMPLING_RATE)
    frame_first_beat[name] = frames[name][0]



## Start all songs from the first beat
for genre in GENRES:
    for song in os.listdir(genre):
        if song[-3:] == 'wav':
            
            _, sr__ = librosa.load(genre +'/' + song, sr=SAMPLING_RATE)
            
            sf.write(genre +'/'+ song, _[frame_first_beat[song[:-4]]:], sr__)
        else:
            print('!! There was a file not ending in .wav. Ignore if this was a .DS_Store file')
            print('This was the file: ', genre+ song)


## Edit frames to reflect we are starting from the first beat
frames_starting_from_first_beat = {}
for song in frames.keys():
    frames_starting_from_first_beat[song] = [_ - frame_first_beat[song] for _ in frames[song]]

## Trim each song to up to 29 seconds. This is because most of the songs have more than 29 seconds, to make the database uniform
for genre in GENRES:
    for song in os.listdir(genre):
        if song[-3:] =='wav':
            _, sr__ = librosa.load(genre +'/'+ song, sr=SAMPLING_RATE)
            
            sf.write(genre +'/'+ song, _[:SAMPLING_RATE*29], SAMPLING_RATE)
        else:
            print('!! There was a file not ending in .wav. Ignore if this was a .DS_Store file')
            print('This was the file: ', song)

## Edit frames to reflect that all the songs are 29 seconds long
frames_ending_at_29_secs = {}
for song in frames_starting_from_first_beat.keys():
    f = np.array(frames_starting_from_first_beat[song])
    frames_ending_at_29_secs[song] = f[f<SAMPLING_RATE*29].copy()


all_mels = {}
mells_train = {}
mells_test = {}

for genre in GENRES:
    num_files = len(os.listdir(genre))
    training_set = set(np.random.choice(list(range(num_files)), 3*num_files//4, False)) 
    for idx, song in enumerate(os.listdir(genre)):
        if song[-3:] =='wav':
            audio, _ = librosa.load(genre +'/'+ song, sr=SAMPLING_RATE)
            S_original_audio = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE, n_mels=128)
            db_spectrogram = librosa.power_to_db(S_original_audio, ref=np.max)
            first_derivative = librosa.feature.delta(db_spectrogram)
            res = np.concatenate((db_spectrogram, first_derivative))

            if res.shape[-1] == 1248:
                all_mels[song[:-4]] = res
                if idx in training_set:
                    mells_train[song[:-4]] = res
                else:
                    mells_test[song[:-4]] = res
            else:
                print('song ' + song+' will be excluded as it is less than 29 seconds')
                print(res.shape)
            

## Convert frames with beat to reflect the shape of mel spectrogram
frames_with_beat = {}
for song in frames_ending_at_29_secs.keys():
    times = librosa.frames_to_time(frames_ending_at_29_secs[song], sr=SAMPLING_RATE)
    new_frames = librosa.time_to_frames(times, sr=MEL_SAMPLING_RATE)
    frames_with_beat[song] = new_frames 

frames_with_beat_encoded = {}
for song in all_mels.keys():
    zeros = np.zeros(all_mels[song].shape[1])
    for beat in frames_with_beat[song]:
        zeros[beat] = 1
    frames_with_beat_encoded[song] = zeros 

del all_mels


os.makedirs(SONGS_LOC+'_train') #Here we'll save all the files used for training
os.makedirs(SONGS_LOC+'_test') #Here we'll save all the files used for test

training_input = np.stack([_ for _ in mells_train.values()])
training_target = np.stack([frames_with_beat_encoded[song] for song in mells_train.keys()])
np.save(SONGS_LOC+'_train/training_target_ballroom.npy', training_target)
del training_target


test_input = np.stack([_ for _ in mells_test.values()])
test_target = np.stack([frames_with_beat_encoded[song] for song in mells_test.keys()])
np.save(SONGS_LOC+'_test/test_target_ballroom.npy', test_target)
del test_target

## Preprocess the data

MIN = -81
MAX = 13

def square(x):
    return x*x

vec_square = np.vectorize(square)

transformed_inputs_test = vec_square((test_input - MIN)/(MAX - MIN))
np.save(SONGS_LOC+'_test/transformed_inputs_test_ballroom.npy', transformed_inputs_test)
del transformed_inputs_test

transformed_inputs = vec_square((training_input - MIN)/(MAX - MIN))
np.save(SONGS_LOC+'_train/transformed_inputs_ballroom.npy', transformed_inputs) 
del transformed_inputs 
  
