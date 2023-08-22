import gc  #garbage collector module

import tensorflow as tf
import models
import custom_losses
import custom_metrics
import parameters as params
import main_post_processing as post_processing

import matplotlib
import matplotlib.pyplot as plt
import plotting_module

import numpy as np
import pandas as pd

import os
import shutil
import pickle

### gets the inputs

transformed_inputs = np.transpose(np.load('songs_train/transformed_inputs.npy'), [0,2,1])[:,:1248,:]
transformed_inputs_test = np.transpose(np.load('songs_test/transformed_inputs_test.npy'), [0,2,1])[:,:1248,:]
transformed_inputs_ballroom = np.transpose(np.load('songs_train/transformed_inputs_ballroom.npy'), [0,2,1])
transformed_inputs_test_ballroom= np.transpose(np.load('songs_test/transformed_inputs_test_ballroom.npy'), [0,2,1])

outputs = np.load('songs_train/training_target.npy')[:,:1248]
outputs_test = np.load('songs_test/test_target.npy')[:,:1248]
outputs_ballroom = np.load('songs_train/training_target_ballroom.npy')
outputs_test_ballroom= np.load('songs_test/test_target_ballroom.npy')

transformed_inputs = np.concatenate([transformed_inputs, transformed_inputs_ballroom], axis=0)
transformed_inputs_test = np.concatenate([transformed_inputs_test, transformed_inputs_test_ballroom], axis=0)
outputs = np.concatenate([outputs, outputs_ballroom], axis=0)
outputs_test = np.concatenate([outputs_test, outputs_test_ballroom], axis=0)

print('shape test input: ',transformed_inputs_test.shape)
print('shape train input: ',transformed_inputs.shape)

optimizer = tf.keras.optimizers.Adam()
loss = custom_losses.weighted_bce

def get_batch(database_X, database_y, batch_size=params.BATCH_SIZE, len_frame=params.LEN_FRAME):
    """
    assumes database_X.shape[:1] == database_y.shape[:1]
    """
    indices = np.random.choice(database_X.shape[0], batch_size, False)
    beginnings = np.random.choice(database_X.shape[1]- len_frame, batch_size, False)
    X = []
    y = []
    for idx, el in enumerate(indices):
        X.append(database_X[el][beginnings[idx]: beginnings[idx] + len_frame])
        y.append(database_y[el][beginnings[idx]: beginnings[idx] + len_frame])

    return np.array(X), np.array(y)

def gradient_step(X, y, my_model, loss_fn=loss, return_loss=True, my_optimizer=optimizer):
    """
    Perform a step of gradient descent updating the loss if past_loss is passed (past_loss != None).
    X,y have to be encoded
    """
    with tf.GradientTape() as tape:
        predictions = my_model(X, training=True)
        my_loss = loss_fn(y, predictions)

    grads = tape.gradient(my_loss, my_model.trainable_variables)
    my_optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
    del tape
    tf.keras.backend.clear_session()
    if return_loss:
        return my_loss

# Metrics for training
past_loss = []
past_loss_test = []
F_scores = []
max_F_score = .5

model = models.bidirectional_model()
model_save = models.bidirectional_model_for_save()
model_save(transformed_inputs[:1])
model(transformed_inputs[:1])

def training_loop(my_model=model, my_optimizer=optimizer, loss_fn=loss,
                  X_train=transformed_inputs, y_train=outputs,
                  X_test=transformed_inputs_test, y_test=outputs_test,
                  beginning=0, epochs=params.EPOCHS, steps_per_epoch=params.STEPS_PER_EPOCH,
                  test_batch_size=params.TEST_BATCH_SIZE, lr_decay=params.DECAY,
                  save_model_at_checkpoint=True, model_for_saving=model_save, max_F_score=max_F_score,
                  past_loss=past_loss, past_loss_test=past_loss_test,
                  F_scores=F_scores):
    """
    Training loop.
    """

    for epoch in range(beginning+1, epochs + 1):
        print('###########################')
        print('Epoch', epoch)
        print('---------------------------')

        ## Gradient step
        for step in range(steps_per_epoch):
            X_batch, y_batch = get_batch(X_train, y_train)
            new_loss = gradient_step(X_batch, y_batch, my_model=my_model,loss_fn=loss_fn,
                                     my_optimizer=my_optimizer)
            past_loss.append(new_loss.numpy())
            del X_batch, y_batch
            #print('past_loss', past_loss)

        ## Test
        predictions = []
        true_results = []
        for test_step in range(steps_per_epoch):
            X_batch_test, y_batch_test = get_batch(X_test, y_test, batch_size=params.TEST_BATCH_SIZE)

            y_pred_test = model(X_batch_test)
            new_loss_test = loss_fn(y_batch_test, y_pred_test)

            #once computed the loss, post process to get more relevan metrics.
            y_pred_test = post_processing.frames_with_beat_batch(y_pred_test)
            past_loss_test.append(new_loss_test.numpy())

            if len(predictions) != 0:
                predictions = np.concatenate((predictions, np.round(y_pred_test).astype(int)))
            else:
                predictions = np.round(y_pred_test).astype(int)

            if len(true_results) == 0:
                true_results = y_batch_test.astype(int)
            else:
                true_results = np.concatenate((true_results, y_batch_test.astype(int)))
            del X_batch_test, y_batch_test, y_pred_test

        optimizer.lr = optimizer.lr*lr_decay

        true_times = custom_metrics.from_frames_to_times_batch(true_results)
        predicted_times = custom_metrics.from_frames_to_ds_times_batch(predictions)
        F_score = custom_metrics.batched_average_F_score(true_times, predicted_times)
        F_scores.append(F_score)

        if epoch%10 == 0:
            ## Plot loss
            os.makedirs('epoch_'+str(epoch))
            loss1 = pd.DataFrame(past_loss, columns = ['train loss'])
            loss2 = pd.DataFrame(past_loss_test, columns = ['test loss'])

            loss_df = loss1.join(loss2)
            loss_df.plot(figsize = (18,12))
            plt.savefig('epoch_'+str(epoch)+'/loss.png')
            plt.close()

            rolling_loss = loss_df.rolling(window=50).mean().dropna()
            rolling_loss.columns = ['rolling loss train', 'rolling loss test']
            rolling_loss.plot(figsize = (18,12))
            plt.savefig('epoch_'+str(epoch)+'/rolling_loss.png')
            plt.close()
            del loss1, loss2, loss_df, rolling_loss

            ## Metrics ##
            cm = custom_metrics.batched_average_cm(true_times, predicted_times)
            del true_times, predicted_times

            plotting_module.plot_cm(cm, my_labels=['Not beat','Beat'], title='Confusion matrix with F-score '+ str(F_score),
                                savefig=True,name_saved_fig='epoch_'+str(epoch)+'/confusion_matrix.png',
                                show=False)
            plt.close()

            F_score_ds = pd.DataFrame(F_scores, columns=['F-scores'])
            F_score_ds.plot(figsize = (18,12))
            plt.savefig('epoch_'+str(epoch)+'/F_score.png')
            plt.close()
            del F_score_ds

            if epoch > 20:
                rolling_F_score_ds = pd.DataFrame(F_scores, columns=['rolling F-scores']).rolling(15).mean()
                rolling_F_score_ds.plot(figsize = (18,12))
                plt.savefig('epoch_'+str(epoch)+'/F_score_rolling.png')
                plt.close()
                del rolling_F_score_ds

            ## Checkpoints
            if save_model_at_checkpoint:
                if epoch%50 == 0:
                    print('Saving model at checkpoint')
                    models.save_weights(my_model, model_for_saving, 'model_epoch_'+str(epoch))
                    os.makedirs('data_at_epoch_'+str(epoch))
                    with open('data_at_epoch_'+str(epoch)+'/past_loss.pkl', 'wb') as f:
                        pickle.dump(past_loss, f)
                        f.close()
                    with open('data_at_epoch_'+str(epoch)+'/past_loss_test.pkl', 'wb') as f:
                        pickle.dump(past_loss_test, f)
                        f.close()
                    with open('data_at_epoch_'+str(epoch)+'/F_scores.pkl', 'wb') as f:
                        pickle.dump(F_scores, f)
                        f.close()
            
            plt.close('all')
            gc.collect()

            if epoch>5 and (epoch-10)%50 !=0:
                shutil.rmtree('epoch_'+str(epoch-10))
