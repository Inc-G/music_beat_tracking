import tensorflow as tf

import models
import custom_losses
import custom_metrics
import parameters as params

import matplotlib
import matplotlib.pyplot as plt
import plotting_module

import numpy as np
import pandas as pd

import os
import shutil
import pickle


### gets the inputs
with open('shuffled_indices.npy', 'rb') as f:
    indices = np.load(f)
    f.close()

indices = np.array(indices)
len_indices = indices.shape[0]
val_idx = np.array(list(range(len_indices)))[indices<(len_indices//5)] # 20% is for validation
train_idx = np.array(list(range(len_indices)))[indices>=(len_indices//5)] # 80% is for training


all_transformed_inputs = np.load('songs_train/transformed_inputs.npy')

print('len validation test: ', val_idx.shape)

transformed_inputs = all_transformed_inputs[train_idx,:,:]
transformed_inputs_test = all_transformed_inputs[val_idx,:,:] 

transformed_inputs = np.transpose(transformed_inputs, [0,2,1])
transformed_inputs_test = np.transpose(transformed_inputs_test, [0,2,1])

all_outputs = np.load('songs_train/training_target.npy')
outputs = all_outputs[train_idx,:]
outputs_test = all_outputs[val_idx,:] 

print('shape validation test input: ',transformed_inputs_test.shape)
print('shape train test input: ',transformed_inputs.shape)




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
        predictions = my_model(X)
        my_loss = loss_fn(y, predictions)

    grads = tape.gradient(my_loss, my_model.trainable_variables)
    my_optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
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

def training_loop(my_model=model, my_optimizer=optimizer, loss_fn=loss,
                  X_train=transformed_inputs, y_train=outputs,
                  X_test=transformed_inputs_test, y_test=outputs_test,
                  beginning=0, epochs=params.EPOCHS, steps_per_epoch=params.STEPS_PER_EPOCH,
                  test_batch_size=params.TEST_BATCH_SIZE,
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
            #print('past_loss', past_loss)

        ## Test
        predictions = []
        true_results = []
        for test_step in range(steps_per_epoch):
            X_batch_test, y_batch_test = get_batch(X_test, y_test, batch_size=test_batch_size)

            y_pred_test = model(X_batch_test)
            new_loss_test = loss_fn(y_batch_test, y_pred_test)
            past_loss_test.append(new_loss_test.numpy())

            if len(predictions) != 0:
                predictions = np.concatenate((predictions, np.round(y_pred_test.numpy()).astype(int)))
            else:
                predictions = np.round(y_pred_test.numpy()).astype(int)

            if len(true_results) == 0:
                true_results = y_batch_test.astype(int)
            else:
                true_results = np.concatenate((true_results, y_batch_test.astype(int)))

        os.makedirs('epoch_'+str(epoch))

        ## Plot loss
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

        ## Metrics ##
        true_times = custom_metrics.from_frames_to_times_batch(true_results)
        predicted_times = custom_metrics.from_frames_to_ds_times_batch(predictions)

        cm = custom_metrics.batched_average_cm(true_times, predicted_times)
        F_score = custom_metrics.batched_average_F_score(true_times, predicted_times)

        plotting_module.plot_cm(cm, my_labels=['Not beat','Beat'], title='Confusion matrix with F-score '+ str(F_score),
                                savefig=True,name_saved_fig='epoch_'+str(epoch)+'/confusion_matrix.png',
                                show=False)
        plt.close()

        F_scores.append(F_score)
        pd.DataFrame(F_scores, columns=['F-scores']).plot(figsize = (18,12))
        plt.savefig('epoch_'+str(epoch)+'/F_score.png')
        plt.close()


        ## Checkpoints
        if save_model_at_checkpoint:
            if max_F_score < .8:
                next_treshold = max_F_score + (1-max_F_score)/4
            elif max_F_score < .9:
                next_treshold = max_F_score + (1-max_F_score)/8
            elif max_F_score < .99:
                next_treshold = max_F_score + .01
            else:
                next_treshold = max_F_score + (1-max_F_score)/32


            if next_treshold < F_score:
                max_F_score = F_score
                print('**********************')
                print('New best F-score:', F_score)
                print('**********************')
                models.save_weights(my_model, model_for_saving, 'F_score'+str(F_score))
            if epoch%25 == 0:
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

        del true_times
        del predicted_times

        if (epoch - 2)%25 != 0 and epoch>1:
            shutil.rmtree('epoch_'+str(epoch-1))
