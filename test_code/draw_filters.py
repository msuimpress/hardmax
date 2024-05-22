import numpy as np
import cv2

FILTER_ADDRESS = "..\\filters\\max_threshold_custom_cbgr_16_filter.npy"
C = 4

def bayer_filter(filter_size):
    bayer = np.array([[[1,0,0],[0,1,0]],
                      [[0,1,0],[0,0,1]]])
    size = round(filter_size/2)
    bayer = np.tile(bayer, (size,size,1))
    return bayer

def lukac_filter(filter_size):
    lukac = np.array([[[0,1,0],[0,0,1]],
                      [[0,1,0],[1,0,0]],
                      [[0,0,1],[0,1,0]],
                      [[1,0,0],[0,1,0]]])
    size = round(filter_size/2)
    lukac = np.tile(lukac, (int(size/2),size,1))
    return lukac

def rgbw_filter(filter_size):
    rgbw = np.array([[[1,0,0,0],[0,1,0,0]],
                      [[0,0,0,1],[0,0,1,0]]])
    size = round(filter_size/2)
    rgbw = np.tile(rgbw, (size,size,1))
    return rgbw

def cfz_filter(filter_size):
    cfz = np.asarray([[[0,1,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]],
                      [[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0]],
                      [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
                      [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]])
    size = round(filter_size/4)
    cfz = np.tile(cfz, (size,size,1))
    return cfz

def learned_filter(filter_address):
    learned_filter = np.load(filter_address)[0]
    return learned_filter

binary_filter = learned_filter(FILTER_ADDRESS)

if C == 3:
    binary_filter_b = np.expand_dims(binary_filter[:,:,0], axis=2)
    binary_filter_g = np.expand_dims(binary_filter[:,:,1], axis=2)
    binary_filter_r = np.expand_dims(binary_filter[:,:,2], axis=2)
    
    binary_filter_visualized = 255*np.concatenate((binary_filter_b,binary_filter_g,binary_filter_r), axis=2)
elif C == 4:
    binary_filter_c = np.tile(np.expand_dims(binary_filter[:,:,0], axis=2),(1,1,3))
    binary_filter_b = np.expand_dims(binary_filter[:,:,1], axis=2)
    binary_filter_g = np.expand_dims(binary_filter[:,:,2], axis=2)
    binary_filter_r = np.expand_dims(binary_filter[:,:,3], axis=2)
    
    binary_filter_visualized = 255*np.logical_or(binary_filter_c, 
                                                 np.concatenate((binary_filter_b,binary_filter_g,binary_filter_r), axis=2))

cv2_learned_filter = cv2.resize(binary_filter_visualized.astype("uint8"), (560,560), interpolation=cv2.INTER_AREA).astype("uint8")
cv2_learned_filter = np.concatenate([np.pad(cv2_learned_filter[:,:,0], ((2, 2),(2, 2)), 'constant')[:,:,np.newaxis],
     np.pad(cv2_learned_filter[:,:,1], ((2, 2),(2, 2)), 'constant')[:,:,np.newaxis],
     np.pad(cv2_learned_filter[:,:,2], ((2, 2),(2, 2)), 'constant')[:,:,np.newaxis]], axis=2)
cv2.imshow("Learned Filter", cv2_learned_filter)
cv2.waitKey(0)

cv2.imwrite("filter_mt_custom_cbgr_16.png", cv2_learned_filter)