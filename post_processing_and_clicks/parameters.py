## preprocessing variables
SAMPLING_RATE = 22016
MEL_SAMPLING_RATE = int(SAMPLING_RATE/512)

## for min-max scaler
MIN = -81
MAX = 13

## model variables
NUM_GRU_LAYERS =10
GRU_WIDTH = 20

## custom loss variables
TOTAL_1_BEATS = 33183
TOTAL_FRAMES = 653952
TOTAL_0_BEATS = 620769

## custom metrics variables
NUM_SECONDS = 10
LEN_FRAME = MEL_SAMPLING_RATE*NUM_SECONDS
WINDOW = .07 #for metrics and downsample

## training variables 
EPOCHS = 300
BATCH_SIZE = 64
STEPS_PER_EPOCH = 60
TEST_BATCH_SIZE = 20
DECAY = .99

## postprocessing variables
LIKELY_BPM = 120
SHIFT = 3


## postprocessing variables
LIKELY_BPM = 190
SHIFT = 3
