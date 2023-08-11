import librosa 

import numpy as np
import pandas as pd
import scipy

import parameters as params 

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

def gaussian(x, mean=0):
    """
    auxiliar
    """
    val = (-(x - mean)**2)/2
    return np.exp(val)

cut = 60/params.LIKELY_BPM
frame_cut = params.MEL_SAMPLING_RATE*cut

curve = np.exp(-1/8 * np.log2(np.arange(0,40,1)/frame_cut)**2)
curve = curve/curve.sum() #curve to weight the correlation. Works as a prior on the tempo, see weighted_correlation below.

#pd.DataFrame(curve).plot(title='weight for the correlation with an emphasis at '+str(params.LIKELY_BPM)+' bpm')
#plt.show()


def weighted_correlation(predictions, len_frame=params.LEN_FRAME, shift=params.SHIFT):
    """
    predictions: tf.tensor of shape [1, len_song]
    len_frame: int. length of the window where the time should stay constant
    shift: int. 
    
    Gets the renormalized self correlation of the predictions.
    """
    beginning = (params.NUM_SECONDS - 1)*len_frame//params.NUM_SECONDS
    end = (params.NUM_SECONDS + 1)*len_frame//params.NUM_SECONDS
    
    cor = np.correlate(predictions.numpy()[0,:len_frame],
                       predictions.numpy()[0,:len_frame],
                       'full')[beginning:end]
    
    second_half = cor[params.MEL_SAMPLING_RATE + shift:]*curve
    return second_half/second_half.sum()

def get_a_beat(predictions, w_cor):
    """
    predictions: tf.tensor of shape [1, len_song]
    w_cor: np.array, the ouput of w_cor = weighted_correlation(predictions).
    
    Gets a beat by convolving the predictions with the curved correlation (the output of weighted_correlation), adding
    the predictions, and taking argmax.
    """
    beat_detected = np.argmax(2*predictions.numpy()[0] + np.convolve(predictions.numpy()[0], w_cor, 'same'))
    return beat_detected


def prob_beat(mode, mel_sampling_rate=params.MEL_SAMPLING_RATE):
    """
    auxiliary function for find_prob_distribution_of_a_beat.

    mode: float.

    Unnormalized probability distribution with given mode over the integers from 0 to mel_sampling_rate.
    The probability distribution is the log2 of a gaussian. 
    
    It is needed to find the next beat, and log2 is there as the probability of having a beat after (2^n)t times
    should be the same as the probability of having a beat after t/(2^n) times
    """
    vals = np.arange(1, mel_sampling_rate)
    return np.concatenate([np.zeros(1),gaussian(np.log2(vals/mode))])

def find_prob_distribution_of_a_beat(w_cor, shift=params.SHIFT, constant_tempo=True, plot=False,):
    """
    w_corr: the output of weighted_correlation
    Returns: np.1darray
    
    Given the weighted corelation, first we find its first peak (actual_peak) that is greater than 2/3 of its
    next peaks. This should be the tempo of the song.
    
    Returns an (unnormalized) probability distribution (namely prob_beat(mode=actual_peak)) on actual_peak*3//2 + 1 frames.
    This is the probability that the frame i has a beat, given that frame 0 has a beat.
    """
    w_cor = np.array([0]*shift + list(w_cor))
    
    peaks, _ = scipy.signal.find_peaks(w_cor)
    
    # Find its first peak (actual_peak) that is greater than 2/3 of its next peaks.
    # This is because, for example, if the tempo is 240 bpms, then on the wcorrelation there will be 2
    # peaks with similar values: one which corresponds to 240 bpms, one which corresponds to 120. Getting the first peak
    # gurantees, in the example with 240 bpms, that the tempo we get is at 240 and not 120. 
    actual_peak = peaks[0]
    for p in peaks:
        if w_cor[actual_peak] < 2*w_cor[p]/3:
            actual_peak = p
            
    if plot:
        pd.DataFrame(w_cor).plot(title='weighted correlation and tempo')
        plt.axvline(actual_peak)
        plt.show()
        
        pd.DataFrame(prob_beat(actual_peak)[:(actual_peak*3)//2 + 1]).plot(title='prob distribution')
        plt.show()
    
    if constant_tempo:
        curve = prob_beat(actual_peak)[:(actual_peak*3)//2 + 1]
        res = []
        for idx, el in enumerate(curve):
            if abs(idx - actual_peak) <= 3:
                res.append(el)
            else:
                res.append(0)
        return np.array(res)
    else:
        return prob_beat(actual_peak)[:(actual_peak*3)//2 + 1]


def search_after(predictions, predicted_beat, prob_distribution):
    """
    predictions: np.1d array
    predicted_beat: int
    prob_distribution: np.1darray
    
    
    Searches for beats after the predicted beat.
    """
    current_beat = predicted_beat
    result = [current_beat]
    
    while current_beat + len(prob_distribution) <= len(predictions):
        next_predictions = predictions[current_beat: current_beat + len(prob_distribution)]
        next_weighted_predictions = next_predictions*prob_distribution
        next_beat = np.argmax(next_weighted_predictions)
        result.append(current_beat + next_beat)
        current_beat += next_beat 
    return result

def search_before(predictions, predicted_beat, prob_distribution):
    """
    predictions: np.1d array
    predicted_beat: int
    prob_distribution: np.1darray
    
    
    Searches for beats before the predicted beat.
    """
    current_beat = predicted_beat
    result = [current_beat]
    
    while current_beat - len(prob_distribution) >=0:
        prev_predictions = predictions[current_beat - len(prob_distribution):current_beat]
        prev_w_predictions = prev_predictions*prob_distribution[::-1]
        prev_beat = np.argmax(prev_w_predictions)
        result.append(current_beat - len(prob_distribution) + prev_beat)
        current_beat -= len(prob_distribution) - prev_beat 
    return result

def frames_with_beat(predictions, constant_tempo=True, plot=False):
    """
    predictions: tf.tensor of shape [1, len_song]. The output of the neural network
    
    returns: a list of length LEN_FRAME with 1 at frame i iff there is a beat at frame i
    """  
    w_cor = weighted_correlation(predictions)
    prob_distribution = find_prob_distribution_of_a_beat(w_cor, constant_tempo=constant_tempo, plot=plot)
    
    single_beat = get_a_beat(predictions, w_cor)
    
    if plot:
        pd.DataFrame(predictions.numpy()[0]).plot(title='predictions with a bit')
        plt.axvline(single_beat, color='red')
        plt.show()
    
    beats_after = search_after(predictions.numpy()[0], single_beat, prob_distribution)
    beats_before = search_before(predictions.numpy()[0], single_beat, prob_distribution) 
    
    all_beats = list(beats_before) + list(beats_after)

    beats_in_frames = np.zeros(predictions.shape[1])
    for _ in all_beats:
        if _ < predictions.shape[1]:
            beats_in_frames[_] = 1
    return beats_in_frames
