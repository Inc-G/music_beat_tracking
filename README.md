# music_beat_tracking

This is a project where I try to have a beat tracking algorithm for music.

Parts of it are heavily inspired by the 2007 paper "Beat Tracking by Dynamic Programming" by Daniel P.W. Ellis, in "Journal of New Music Research"; and by the 2011 paper "Enhanced beat tracking with context-aware neural networks" by S Böck, M Schedl, in Proc. Int. Conf. Digital Audio Effects.


The project, roughly speaking, is divided into three parts:

(1) pre processing. The notebook I used is in the folder pre_processing.

Essentially, after uniformizing a bit the dataset, each song is sampled with sampling rate 22016, then I take the mel spectrogram of it, converted to decibels, and
appended to the result of the mel spectrogram its first order difference. 

(2) training. I train a neural network with 4 bidirectional gru layers and a dense layer at the end. I have to take a weighted loss (in the module custom_losses) to compensate the imbalanced dataset. Similarly the metrics are customized for the task of beat tracking (you can see them in custom_metrics).

(3) from the output of the neural network to the beats. This is in the post_processing_and_clicks folder. To go from the outcome of the neural network to the predicted beats, I do the following:

- Find (a probability distribution for) the tempo of the song. This is done, roughly speaking, by finding the auto-correlations of the outcomes of the neural network. The relevant function is find_prob_distribution_of_a_beat.
- Find one beat in the song; the relevant function is get_a_beat.
- Find the beats after and before the found beat. To find the beat after a beat x, I take the element-wise product of the a probability distribution for a tempo * outcomes of the nn, and find a peak. The relevant functions are search_after and search_before.

There is also a function (namely, add_clicks) in add_clicks.py that adds the clicks to a song.

# Relevant references for preprocessing:

I used the database and annotations from https://github.com/CPJKU/BallroomAnnotations

This paper should be relevant too: 
F. Krebs, S. Böck, and G. Widmer. Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio. Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR), Curitiba, Brazil, 2013.  
