# music_beat_tracking
## Usage
Install all the packages in the .yml file beat_tracking.yml, go to the folder flask_webapp and run webapp_beat_tracking.py. After uploading your piece of music,  if you click "add beats" it will add beats to it.

The piece of music you upload should have constant tempo.

## Examples
You can find some examples of end results in flask_webapp/static. Below I report three of them, and an example of the webapp.

https://github.com/user-attachments/assets/c7128fce-b91d-4912-991d-5faa0b0b2af1

https://github.com/Inc-G/music_beat_tracking/assets/55004390/cdde0f8c-ee8d-466b-b621-cdbbf305e068

https://github.com/Inc-G/music_beat_tracking/assets/55004390/9f1f1de4-e88e-4469-aa19-d06c826413fc

https://github.com/Inc-G/music_beat_tracking/assets/55004390/2b07520d-6d63-4fe9-b982-16374560f454


## On this project

In this project I created a beat tracking algorithm for music.

Parts of it are heavily inspired by the 2007 paper "Beat Tracking by Dynamic Programming" by Daniel P.W. Ellis, in "Journal of New Music Research"; and by the 2011 paper "Enhanced beat tracking with context-aware neural networks" by S Böck, M Schedl, in Proc. Int. Conf. Digital Audio Effects.


The project, roughly speaking, is divided into three parts:

(1) pre processing. The notebook I used is in the folder pre_processing.

The two scripts preprocessing_GTZAN.py and preprocessing_ballroom.py are the relevant ones.

Essentially, after exploring and uniformizing a bit the dataset, each song is sampled with sampling rate 22016, then I take the mel spectrogram of it, converted to decibels, and appended to the result of the mel spectrogram its first order difference. Each song in the ballroom dataset is first cut at the beginning so that it starts from the first beat, and at the end so that it lasts 29 seconds. This is to make the dataset uniform, as there are songs which have their first beat late.

(2) training. I train a neural network with 5 bidirectional gru layers, followed by 5 pairs (bidirectional gru, dropout layer) and a dense layer at the end. I have to take a weighted loss (in the module custom_losses) to compensate the imbalanced dataset, and the decay for the learning rate is .99 (applied at each epoch, see training/main.py). Similarly the metrics are customized for the task of beat tracking (you can see them in custom_metrics). 

The stream of information is: sample a batch of songs, sample 10 consecutive seconds from each song in the batch, feed the resulting batch of 10 seconds of song to the neural network to predict the beats.

As the model is trained on tracks of 10 seconds, the best performance is for such tracks.

I trained it for 300 epochs, see the metrics at training/metrics_at_epoch_300. The results of the metrics are after post-processing the predictions of the neural network.

(3) from the output of the neural network to the beats. This is in the post_processing_and_clicks folder. To go from the outcome of the neural network to the predicted beats, I do the following:

- Find (a probability distribution for) the tempo of the song. This is done, roughly speaking, by finding the auto-correlations of the outcomes of the neural network. The relevant function is find_prob_distribution_of_a_beat.
- Find one beat in the song; the relevant function is get_a_beat.
- Find the beats after and before the found beat. To find the beat after a beat x, I take the element-wise product of the a probability distribution for a tempo * outcomes of the nn, and find a peak. The relevant functions are search_after and search_before.

There is also a function (namely, add_clicks) in add_clicks.py that adds the clicks to a song.

# Datasets:

I used two datasets.

The GTZAN dataset which you can download here https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
I used these annotations http://anasynth.ircam.fr/home/english/media/GTZAN-rhythm

The ballroom dataset and annotations from https://github.com/CPJKU/BallroomAnnotations

These papers should be relevant: 
- F. Krebs, S. Böck, and G. Widmer. Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio. Proceedings of the 14th International Society for Music Information Retrieval Conference (ISMIR), Curitiba, Brazil, 2013.  

- U. Marchand, Q. Fresnel and G. Peeters, "GTZAN-Rhythm: extending the GTZAN test-set with beat, downbeat and swing annotations", in ISMIR 2015 Late-Breaking Session, Malaga, Spain

# To replicate the results:

Download the two datasets above, rename the GTZAN dataset 'GTZAN', and run the scripts for preprocessing in the same folders where there are the datasets. This will generate 8 .npy files; 4 of which from GTZAN and 4 from the ballroom dataset. Those are (train_inputs, train_outputs, test_inputs, test_outputs) for each dataset.

Train the neural network (on colab or using a GPU, or lots of patience) using the script main.py in the training folder, and the 8 files from the previous step.

