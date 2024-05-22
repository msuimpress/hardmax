import tensorflow as tf
import numpy as np
import cv2
import os
from math import log10
from skimage.metrics import structural_similarity as find_ssim

# Initializating the addresses
MODEL_ADDRESS = "..\\models\\cfa_demosaicer_cfz_8"
IMAGE_ADDRESS = "..\\dataset\\test_images_kodak"
FILTER_ADDRESS = "..\\learned_filters\\test_images_kodak"
FILTER_NAME = "Bayer"

# Initializating the constants
NORMALIZE = False
C = 3
P = 8
BATCH_SIZE = 128
IMAGE_SIZE = 3*P

# Calculates PSNR value between two images.
def find_psnr(im1, im2):
    return 20*log10(255/np.sqrt(np.square(im1 - im2).mean()))
# Divides an array of images into smaller blocks.
def divide_to_blocks(image, stride=8, image_size=24):
    image_blocks = []
    height, width = image.shape[:2]
    # Calculating the number of blocks per height and width
    block_no_vertical = ((height - image_size) // stride) + 1
    block_no_horizontal = ((width - image_size) // stride) + 1
    # Dividing the image into blocks
    for i in range(block_no_vertical):
        for j in range(block_no_horizontal):
            image_block = image[i*stride:(i*stride) + image_size, j * stride:(j * stride) + image_size]
            image_blocks.append(image_block)
    return image_blocks
# Merges image blocks to create a full image.
def merge_blocks(blocks, height, width):
    construction = np.array([])
    for i in range(height):
        const_width = blocks[width*i]
        for j in range(width-1):
            const_width = np.concatenate((const_width, blocks[width*i+j+1]), 1)
        if i == 0:
            construction = const_width
        else:
            construction = np.concatenate((construction, const_width), 0)
    return construction

if FILTER_NAME == "Bayer":       # 3P x 3P BGGR Bayer Filter
    bayer_filter = np.asarray([[[1,0,0],[0,1,0]],
                               [[0,1,0],[0,0,1]],])
    cfa_filter = np.tile(bayer_filter,(int(IMAGE_SIZE/2),int(IMAGE_SIZE/2),1))
elif FILTER_NAME == "CFZ":       # 3P x 3P CBGR CFZ Filter
    cfz_filter = np.asarray([[[0,1,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]],
                               [[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0]],
                               [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
                               [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]])
    cfa_filter = np.tile(cfz_filter,(int(IMAGE_SIZE/4),int(IMAGE_SIZE/4),1))
elif FILTER_NAME == "Fixed":    # 3P x 3P Maximum Thresholding Filter
    fixed_filter = np.load(FILTER_ADDRESS)[0]
    cfa_filter = np.tile(fixed_filter,(3,3,1))

# Loading the model
model = tf.keras.models.load_model(MODEL_ADDRESS)
original_images = []
predicted_images = []
image_psnr = []
image_ssim = []
for root, dirs, files in os.walk(IMAGE_ADDRESS, topdown=True):
    for name in files:
        print("Image name: ", name)
        new_image = cv2.imread(os.path.join(IMAGE_ADDRESS, name))
        if C == 4:
            image_grayscale = np.expand_dims(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY), axis=2)
            new_image = np.concatenate((image_grayscale, new_image), axis=2)
        image_blocks = np.asarray(divide_to_blocks(new_image,P,IMAGE_SIZE))
        
        height, width = new_image.shape[:2]
        height = ((height - IMAGE_SIZE) // P) + 1
        width = ((width - IMAGE_SIZE) // P) + 1
        
        image_input = []
        for block in image_blocks:
            block_filtered = np.multiply(block, cfa_filter)
            block_mask = np.expand_dims(np.sum(block_filtered, axis=2), axis=2)
            image_input.append(block_mask)
        image_input = np.asarray(image_input)
        image_output = image_blocks[:,P:2*P,P:2*P]
        if C == 4: image_output = image_output[:,:,:,1:]
        
        # Merging the original image blocks as one image (this is the original image in the same size as the block strides)
        original_image = merge_blocks(image_output.astype(int), height, width)
        # Normalizing the dataset
        if NORMALIZE == True:
            image_input = image_input / 255
            image_output = image_output / 255

        # Converting the input images and output ground truth into TensorFlow dataset object
        image_dataset = tf.data.Dataset.from_tensor_slices((image_input, image_output)).batch(BATCH_SIZE)
        
        print("Reconstructing the image from the samples.")
        predictions = model.predict(image_dataset, verbose=1)
            
        # Merging the predictions as one image
        predicted_image = merge_blocks((255*predictions).astype(int), height, width)
        predicted_image = np.where(predicted_image > 255,
                    255,
                    predicted_image)
        predicted_image = np.where(predicted_image < 0,
                    0,
                    predicted_image)
        
        original_images.append(original_image)
        predicted_images.append(predicted_image)
        
        image_ssim.append(find_ssim(original_image, predicted_image, data_range=255, multichannel=True))
        image_psnr.append(find_psnr(original_image, predicted_image))

for i in range(len(image_psnr)):
    print(image_psnr[i])
for i in range(len(image_psnr)):
    print(image_ssim[i])
"""
# Visualizing the original and the predicted images
index = 4

# Visualizing the original image
org_img = original_images[index].astype("uint8")
cv2_original_image = cv2.resize(org_img, (2*org_img.shape[1],2*org_img.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("Original Image", cv2_original_image)
cv2.waitKey(0)
#cv2.imwrite("4_original_image.png", predicted_images[index])

# Visualizing the predicted image
pred_img = predicted_images[index].astype("uint8")
cv2_predicted_image = cv2.resize(pred_img, (2*pred_img.shape[1],2*pred_img.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("Predicted Image", cv2_predicted_image)
cv2.waitKey(0)
#cv2.imwrite("4_bayer_predicted_image.png", predicted_images[index])

print("\nImage PSNR: ", find_psnr(original_images[index], predicted_images[index]))
"""