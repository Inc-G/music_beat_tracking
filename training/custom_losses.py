"""
There are two losses. weighted_bce to be used with imbalanced classes, and target_change changes the target variable, smoothing it for early epochs.
"""
import tensorflow as tf
import numpy as np

import parameters as params


weight_for_0 = (1 / params.TOTAL_0_BEATS) * (params.TOTAL_FRAMES / 2.0)
weight_for_1 = (1 / params.TOTAL_1_BEATS) * (params.TOTAL_FRAMES / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

## Loss with imbalanced classes

usual_bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

def weighted_bce(y_true, y_pred, weights=class_weight):
    """
    Weighted bce, where the weights on the samples where a beat happens are with weight class_weight[1],
    and those where there is no beat are with class_weight[0].
    """
    y_true =  tf.convert_to_tensor(y_true, dtype='float32')
    y_t_flat = tf.reshape(y_true, [y_true.shape[0]*y_true.shape[1],1])
    y_p_flat = tf.reshape(y_pred, [y_true.shape[0]*y_true.shape[1],1])
    unweighted_loss = usual_bce(y_t_flat, y_p_flat)
    weights = tf.reshape(y_t_flat*(weights[1] - weights[0]) + weights[0], y_t_flat.shape[0])
    res = tf.reduce_mean(unweighted_loss*weights)
    
    # Delete a bunch of variables to free space - needed to train on Colab
    del y_t_flat
    del y_p_flat
    del unweighted_loss
    del weights
    return res

## Change target to measure more accurately the distance from a beat

# what follows has not been used (yet)

def distance_from_1_auxiliary(array):
    """
    Auxiliary function
    """
    res = [0 for _ in array]
    last_1 = -10
    for idx, el in enumerate(array):
        if el==1:
            res[idx]+=1
            last_1 = idx
        else:
            res[idx]+=1/(1+idx-last_1)
    return res

def distance_from_1(array):
    """
    Modifies target variable to penalize less if the model misses a beat by not much.
    Example: [1,0,0,1,0,0,0,1,0,0] --> [1, .5, .5, 1, .5, 0.3333, .5, 1, .5, 0.3333]
    """
    left_to_right = distance_from_1_auxiliary(array)
    right_to_left = distance_from_1_auxiliary(array[::-1])
    return np.maximum(np.array(left_to_right),np.array(right_to_left[::-1]))

def modify_target(target):
    """
    Modifies each batch of the target variable applying distance_from_1 function
    """
    return np.apply_along_axis(distance_from_1, 1, target)

def target_change(true_target, modified_target, epoch):
    shift = epoch//20
    return (modified_target + shift*true_target)/(shift + 1)
