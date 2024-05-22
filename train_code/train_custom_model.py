import tensorflow as tf
import numpy as np
from os.path import exists
from os import makedirs
from math import ceil
import joint_models as cfa
import cfa_models as fil

# Model Hyperparameters
CFA_FILTER = "MaxThreshold"
P = 8       # Filter Size
C = 3       # Number of channel
K = 3*P     # Number of proposals
F = 128     # Reconstruction network filter size

#IMAGE_ADDRESS = "..\\dataset\\training_blocks_" + str(P) + ".npy"   # Initializing the address
IMAGE_ADDRESS = "..\\dataset\\training_blocks_8_1m.npy"   # Initializing the address
FILTER_ADDRESS = "..\\filters\\max_threshold_custom_cbgr_8_filter.npy"
IMAGE_HEIGHT = 3*P
IMAGE_WIDTH = 3*P

# Training parameters
NORMALIZE = False
BATCH_SIZE = 128
EPOCH = 100
LR = 0.001

#with tf.device("GPU:0"):
# Loading and preparing the training dataset
if C == 3:
    training_input = np.load(IMAGE_ADDRESS)[:500000,:,:,1:]
    #data_file = np.load(IMAGE_ADDRESS)
    #training_input = data_file.f.arr_0[:,:,:,1:]
    training_input_num = training_input.shape[0]
    training_output = training_input
elif C == 4:
    training_input = np.load(IMAGE_ADDRESS)[:500000,:,:,:]
    #data_file = np.load(IMAGE_ADDRESS)
    #training_input = data_file.f.arr_0
    training_input_num = training_input.shape[0]
    training_output = training_input[:,:,:,1:]

# Normalizing the dataset
if NORMALIZE: 
    training_input = (training_input / 255).astype("float32")
    training_output = (training_output / 255).astype("float32")
    
print("Input Shape: ", training_input.shape)
print("Output Shape: ", training_output.shape)

# Initializing the model
if CFA_FILTER == "WeightedSoftmax":
    INITIAL_ALPHA = 1
    GAMMA = 2.5*10**(-5)
    [model, callbacks] = cfa.WeightedSoftmax_Custom(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                   channel=C, sm_filter_size=P, 
                                                   gamma=GAMMA, batch_size=BATCH_SIZE, 
                                                   input_num=training_input_num)
elif CFA_FILTER == "MaxThreshold":
    [model, callbacks] = cfa.MaxThreshold_Custom(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                 channel=C, filter_size=P)
elif CFA_FILTER == "Bayer":
    bayer_filter = fil.bayer_filter(3*P)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, bayer_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    [model, callbacks] = cfa.FixedFilter_Custom(IMAGE_HEIGHT, IMAGE_WIDTH)
    
elif CFA_FILTER == "Lukac":
    lukac_filter = fil.lukac_filter(3*P)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, lukac_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    [model, callbacks] = cfa.FixedFilter_Custom(IMAGE_HEIGHT, IMAGE_WIDTH)

elif CFA_FILTER == "RGBW":
    rgbw_filter = fil.rgbw_filter(3*P)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, rgbw_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    [model, callbacks] = cfa.FixedFilter_Custom(IMAGE_HEIGHT, IMAGE_WIDTH)

elif CFA_FILTER == "CFZ":
    cfz_filter = fil.cfz_filter(3*P)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, cfz_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    [model, callbacks] = cfa.FixedFilter_Custom(IMAGE_HEIGHT, IMAGE_WIDTH)

elif CFA_FILTER == "Fixed":
    custom_filter = fil.custom_filter(FILTER_ADDRESS)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, custom_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input) 
    [model, callbacks] = cfa.FixedFilter_Custom(IMAGE_HEIGHT, IMAGE_WIDTH)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mae"])

# Training the model
history = model.fit(x=training_input,
                    y=training_output,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=ceil(training_input_num/BATCH_SIZE),
                    callbacks=callbacks)

if not exists(".\\models"):
    makedirs(".\\models")

MODEL_NAME = ".\\models\\cfa_demosaicer"
if CFA_FILTER == "MaxThreshold":
    MODEL_NAME = MODEL_NAME + "_max_threshold"
elif CFA_FILTER == "WeightedSoftmax":
    MODEL_NAME = MODEL_NAME + "_weighted_softmax"
elif CFA_FILTER == "Fixed":
    MODEL_NAME = MODEL_NAME + "_fixed"
elif CFA_FILTER == "Bayer":
    MODEL_NAME = MODEL_NAME + "_bayer"
elif CFA_FILTER == "Lukac":
    MODEL_NAME = MODEL_NAME + "_lukac"
elif CFA_FILTER == "RGBW":
    MODEL_NAME = MODEL_NAME + "_RGBW"
elif CFA_FILTER == "CFZ":
    MODEL_NAME = MODEL_NAME + "_CFZ"
MODEL_NAME = MODEL_NAME + "_custom"
if C == 3:
    MODEL_NAME = MODEL_NAME + "_bgr"
elif C == 4:
    MODEL_NAME = MODEL_NAME + "_cbgr"
#MODEL_NAME = MODEL_NAME + "_" + str(P)
MODEL_NAME = MODEL_NAME + "_500k_5"
model.save(MODEL_NAME)

print(history.history)