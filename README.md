# music_beat_tracking
## Usage
Install all the packages in the .yml file beat_tracking.yml, go to the folder flask_webapp and run webapp_beat_tracking.py. After uploading your own piece of music,  if you click "add beats" it will add beats to your piece of music.

The piece of music you upload should have constant tempo.

## Examples
You can find some examples of end results in flask_webapp/static. Below I report three of them, and an example of the webapp.

https://github.com/Inc-G/music_beat_tracking/assets/55004390/cb74a6a5-a679-45b5-9ecf-173977eeaf76

https://github.com/Inc-G/music_beat_tracking/assets/55004390/2bd3c48e-e45b-4463-b780-06fd688a9918

https://github.com/Inc-G/music_beat_tracking/assets/55004390/497db2f2-643c-47f2-9195-c44ff20de00b

https://github.com/Inc-G/music_beat_tracking/assets/55004390/670ef95a-0fdc-4b54-ab2b-db1fd0d02fa1

## On this project

This is a project where I try to have a beat tracking algorithm for music.

Parts of it are heavily inspired by the 2007 paper "Beat Tracking by Dynamic Programming" by Daniel P.W. Ellis, in "Journal of New Music Research"; and by the 2011 paper "Enhanced beat tracking with context-aware neural networks" by S Böck, M Schedl, in Proc. Int. Conf. Digital Audio Effects.


The project, roughly speaking, is divided into three parts:

(1) pre processing. The notebook I used is in the folder pre_processing.

Essentially, after uniformizing a bit the dataset, each song is sampled with sampling rate 22016, then I take the mel spectrogram of it, converted to decibels, and
appended to the result of the mel spectrogram its first order difference. 

(2) training. I train a neural network with 3 bidirectional gru layers and a dense layer at the end. I have to take a weighted loss (in the module custom_losses) to compensate the imbalanced dataset. Similarly the metrics are customized for the task of beat tracking (you can see them in custom_metrics). I tried with 2 or 4 gru layers too, but the best performance seems to be with 3 layers.

I trained it for around 175 epochs, at epoch 125 I changed the learning rate from .001 to .0001; and the top F-score was reached at about epoch 150.

(3) from the output of the neural network to the beats. This is in the post_processing_and_clicks folder. To go from the outcome of the neural network to the predicted beats, I do the following:

- Find (a probability distribution for) the tempo of the song. This is done, roughly speaking, by finding the auto-correlations of the outcomes of the neural network. The relevant function is find_prob_distribution_of_a_beat.
- Find one beat in the song; the relevant function is get_a_beat.
- Find the beats after and before the found beat. To find the beat after a beat x, I take the element-wise product of the a probability distribution for a tempo * outcomes of the nn, and find a peak. The relevant functions are search_after and search_before.

There is also a function (namely, add_clicks) in add_clicks.py that adds the clicks to a song.

# Relevant references for preprocessing:

I used the database and annotations from https://github.com/CPJKU/BallroomAnnotations

This paper should be relevant too: 
F. Krebs, S. Böck, and G. Widmer. Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio. Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR), Curitiba, Brazil, 2013.  
