import tensorflow as tf
import numpy as np
import cv2
import os
from math import log10
from skimage.metrics import structural_similarity as find_ssim

# Initializating the addresses
MODEL_ADDRESS = "..\\models_scenario_2\\cfa_demosaicer_max_threshold_custom_rgb_8"
IMAGE_ADDRESS = "..\\dataset\\test_images_kodak"

# Initializating the constants
NORMALIZE = False
C = 3
P = 12
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
        image_blocks = np.asarray(divide_to_blocks(new_image,IMAGE_SIZE,IMAGE_SIZE))
        
        height, width = new_image.shape[:2]
        height = ((height - IMAGE_SIZE) // P) + 1
        width = ((width - IMAGE_SIZE) // P) + 1
        
        image_input = np.copy(image_blocks)
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
""""""
# Visualizing the original and the predicted images
index = 0

# Visualizing the original image
org_img = original_images[index].astype("uint8")
cv2_original_image = cv2.resize(org_img, (2*org_img.shape[1],2*org_img.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("Original Image", cv2_original_image)
cv2.waitKey(0)

# Visualizing the predicted image
pred_img = predicted_images[index].astype("uint8")
cv2_predicted_image = cv2.resize(pred_img, (2*pred_img.shape[1],2*pred_img.shape[0]), interpolation=cv2.INTER_AREA)
cv2.imshow("Predicted Image", cv2_predicted_image)
cv2.waitKey(0)
#cv2.imwrite("max_threshold_predicted_image.png", predicted_images[index])

print("\nImage PSNR: ", find_psnr(original_images[index], predicted_images[index]))

# Extracting the learned binary filter 
name_filter = model.layers[1].name
learned_filter = model.get_layer(name_filter)
learned_filter_weights = learned_filter.get_weights()[0]
binary_filter = np.zeros_like(learned_filter_weights)
for i in range(P):
    for j in range(P):
        threshold = np.max(learned_filter_weights[:,i,j])
        binary_filter[:,i,j] = tf.where(learned_filter_weights[:,i,j] < threshold,
                    0*tf.ones_like(learned_filter_weights[:,i,j], dtype=tf.float32),
                    1*tf.ones_like(learned_filter_weights[:,i,j], dtype=tf.float32))

# Visualizing the binary filter
if C == 3:
    binary_filter_b = np.expand_dims(binary_filter[0,:,:,0], axis=2)
    binary_filter_g = np.expand_dims(binary_filter[0,:,:,1], axis=2)
    binary_filter_r = np.expand_dims(binary_filter[0,:,:,2], axis=2)
    
    binary_filter_visualized = 255*np.concatenate((binary_filter_b,binary_filter_g,binary_filter_r), axis=2)
elif C == 4:
    binary_filter_c = np.tile(np.expand_dims(binary_filter[0,:,:,0], axis=2),(1,1,3))
    binary_filter_b = np.expand_dims(binary_filter[0,:,:,1], axis=2)
    binary_filter_g = np.expand_dims(binary_filter[0,:,:,2], axis=2)
    binary_filter_r = np.expand_dims(binary_filter[0,:,:,3], axis=2)
    
    binary_filter_visualized = 255*np.logical_or(binary_filter_c, 
                                                 np.concatenate((binary_filter_b,binary_filter_g,binary_filter_r), axis=2))

cv2_learned_filter = cv2.resize(binary_filter_visualized.astype("uint8"), (480,480), interpolation=cv2.INTER_AREA).astype("uint8")
cv2.imshow("Learned Filter", cv2_learned_filter)
cv2.waitKey(0)

cv2.imwrite("max_threshold_filter_bgr_8.png", cv2_learned_filter)

#np.save("BGR_max_threshold_filter", binary_filter)