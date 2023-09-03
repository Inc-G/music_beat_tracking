"""
Module with custom metrics.
"""

import numpy as np

import parameters as params


def from_frames_to_times(frames):
    """
    frames: vector with 1 at frame i iff there is a beat at frame i
    Converts frames which have a beat to the times when the beat happens
    """
    res = []
    for idx, el in enumerate(frames):
        if el == 1:
            res.append(idx)
    res = np.array(res)
    return res/params.MEL_SAMPLING_RATE
            

def downsample(beat_times, window=params.WINDOW):
    """
    beat times: sorted np.array with the times when is a beat
    downsamples the beat_times predicted, so that if the model predicts two
    beats at times which is less than the window, it merges them to a unique beat
    
    speed can be optimized
    """
    if len(beat_times) == 0:
        return np.array([])
    time_slots = np.arange(0, beat_times[-1] + window, window)
    res = []
    for idx, el in enumerate(time_slots[:-1]):
        for _ in beat_times:
            if el<= _ and _ <time_slots[idx+1]:
                res.append(el)
                break
                
    #if there are two beats within WINDOW time, we merges them to the first one. So for example if WINDOW=.1,
    #and there are beats at [.1,.2,.6,.9], we merge 1. and .2 and return [.1,.6,.9]
    spaced_out = [res[0]] 
    for _ in res:
        if abs(_ - spaced_out[-1])>params.WINDOW:
            spaced_out.append(_)
    return  np.array(spaced_out)

def true_positive(truth, predicted, window=params.WINDOW):
    """
    truth: np.array with times in secods with a beat
    predicted: np.array with predicted times in secods with a beat
    window: if two beats fall within window, they count as a beat
    """
    tp = 0
    for p_beat in predicted:
        for t_beat in truth:
            if abs(t_beat - p_beat)< window:
                tp+=1
                break
                
    return tp
    
    
def false_negative(truth, predicted, window=params.WINDOW):
    """
    Computes false negatives
    """
    num_of_beats_detected = true_positive(predicted, truth, window)#notice that predicted and truth are swapped
    num_of_missed_beats = len(truth) - num_of_beats_detected
    return num_of_missed_beats

def F_score(truth, predicted, window=params.WINDOW):
    """
    Assumes that predicted is downsampled
    """
    tp = true_positive(truth, predicted, window)
    if len(predicted) == 0 or tp == 0:
        return 0
    precision = tp/len(predicted)
    recall = tp/len(truth)
    
    return (2*precision*recall)/(precision+recall)

def cm(truth, predicted, window=params.WINDOW):
    """
    Confusion matrix. Assumes that predicted is downsampled
    """
    if len(predicted) == 0:
        return np.array([[0,1],[1,0]])
    
    tp = true_positive(truth, predicted, params.WINDOW)
    fn = false_negative(truth, predicted, params.WINDOW)
    fp = len(predicted) - tp
    tn = params.LEN_FRAME - len(predicted) - fn
    return np.array([[tn/(params.LEN_FRAME-len(predicted)),fn/(params.LEN_FRAME-len(predicted))],[fp/len(predicted),tp/len(predicted)]])
    
def batched_average_cm(truth, predicted, window=params.WINDOW):
    """
    computes the mean cm for a batch of pairs (truth, predicted)
    """
    res = []
    for idx in range(len(truth)):
        res.append(cm(truth[idx], predicted[idx], params.WINDOW))
    res = np.array(res)
    return np.mean(res, axis=0)

def batched_average_F_score(truth, predicted, window=params.WINDOW):
    """
    computes the mean F-score for a batch of pairs (truth, predicted)
    """
    res = []
    for idx in range(len(truth)):
        res.append(F_score(truth[idx], predicted[idx], params.WINDOW))
    res = np.array(res)
    return np.mean(res, axis=0)

def from_frames_to_ds_times_batch(predicted, window=params.WINDOW):
    """
    Applies downsample(from_frames_to_times()) to each entry in the batch of predictions.
    """
    res = []
    for _ in predicted:
        res.append(downsample(from_frames_to_times(_),window=params.WINDOW))
    return res

def from_frames_to_times_batch(truth, window=params.WINDOW):
    """
    Applies from_frames_to_times() to each entry in the batch of truth.
    """
    res = []
    for _ in truth:
        res.append(from_frames_to_times(_))
    return res
