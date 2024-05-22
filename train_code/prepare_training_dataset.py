import cv2
import numpy as np
import os
import random 

IMAGE_ADDRESS_1 = "..\\..\\BSDS500\\images\\test"
IMAGE_ADDRESS_2 = "..\\..\\BSDS500\\images\\train"
IMAGE_ADDRESS_3 = "..\\..\\BSDS500\\images\\val"
TRAINING_DATA_SAVE_ADDRESS = "C:\\Users\\ca1389\\Desktop\\Lab\\Project\\cfa_learning\\dataset\\training_blocks_8"
CFA_SIZE = 8               # The size of the repeating filter block, represented as P in the paper
BLOCK_SIZE = 3*CFA_SIZE     # The size of the image block used in training.

blocks = []
for root, dirs, files in os.walk(IMAGE_ADDRESS_1, topdown=True):
    for name in files:
       print(os.path.join(IMAGE_ADDRESS_1, name))
       # OpenCV reads images in BGR format
       img = cv2.imread(os.path.join(IMAGE_ADDRESS_1, name))
       # Creating the panchromatic channel as the fourth channel to be used in the training.
       # panchroma = np.expand_dims((img[:,:,0].astype("uint16") + img[:,:,1].astype("uint16") + img[:,:,2].astype("uint16")), axis=2)
       grayscale = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2)
       img = np.concatenate((grayscale, img), axis=2)
       
       height, width = img.shape[:2]
       row = int(height/CFA_SIZE)-2
       col = int(width/CFA_SIZE)-2
       for i in range(row):
           for j in range(col):
               blocks.append(img[i*CFA_SIZE:(i+3)*CFA_SIZE,j*CFA_SIZE:(j+3)*CFA_SIZE])
               
for root, dirs, files in os.walk(IMAGE_ADDRESS_2, topdown=True):
    for name in files:
       print(os.path.join(IMAGE_ADDRESS_2, name))
       # OpenCV reads images in BGR format
       img = cv2.imread(os.path.join(IMAGE_ADDRESS_2, name))
       # Creating the panchromatic channel as the fourth channel to be used in the training.
       # panchroma = np.expand_dims((img[:,:,0].astype("uint16") + img[:,:,1].astype("uint16") + img[:,:,2].astype("uint16")), axis=2)
       grayscale = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2)
       img = np.concatenate((grayscale, img), axis=2)
       
       height, width = img.shape[:2]
       row = int(height/CFA_SIZE)-2
       col = int(width/CFA_SIZE)-2
       for i in range(row):
           for j in range(col):
               blocks.append(img[i*CFA_SIZE:(i+3)*CFA_SIZE,j*CFA_SIZE:(j+3)*CFA_SIZE])

for root, dirs, files in os.walk(IMAGE_ADDRESS_3, topdown=True):
    for name in files:
       print(os.path.join(IMAGE_ADDRESS_3, name))
       # OpenCV reads images in BGR format
       img = cv2.imread(os.path.join(IMAGE_ADDRESS_3, name))
       # Creating the panchromatic channel as the fourth channel to be used in the training.
       # panchroma = np.expand_dims((img[:,:,0].astype("uint16") + img[:,:,1].astype("uint16") + img[:,:,2].astype("uint16")), axis=2)
       grayscale = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2)
       img = np.concatenate((grayscale, img), axis=2)
       
       height, width = img.shape[:2]
       row = int(height/CFA_SIZE)-2
       col = int(width/CFA_SIZE)-2
       for i in range(row):
           for j in range(col):
               blocks.append(img[i*CFA_SIZE:(i+3)*CFA_SIZE,j*CFA_SIZE:(j+3)*CFA_SIZE])

blocks = np.asarray(blocks)

# Shuffling the blocks for a randomized training dataset
shuffle_list = list(range(len(blocks)))
random.shuffle(shuffle_list)
blocks = blocks[shuffle_list]

# Saving the model.
if not os.path.exists("C:\\Users\\ca1389\\Desktop\\Lab\\Project\\cfa_learning\\dataset"):
    os.makedirs("C:\\Users\\ca1389\\Desktop\\Lab\\Project\\cfa_learning\\dataset")

np.save(TRAINING_DATA_SAVE_ADDRESS, blocks)

# Visualizing the dataset
index = 5

cv2_img = cv2.resize(blocks[index,:,:,1:].astype("uint8"), (480,480), interpolation=cv2.INTER_AREA)
cv2.imshow("Image Block", cv2_img)
cv2.waitKey(0)

cv2_pan = cv2.resize(blocks[index,:,:,0].astype("uint8"), (480,480), interpolation=cv2.INTER_AREA)
cv2.imshow("Panchromatic Block", cv2_pan)
cv2.waitKey(0)
