import numpy as np
import skimage.io as ski


# input the RGB Data, output the binary mask (0, 1)
def label2binary(data):
    re = np.zeros(data.shape[:-1])
    re[np.all(data==[128,0,0,255], axis=-1)]=1
    return re
# write the binary to tiff file
def write2tif(fname, data):
    data = np.uint8(data)
    ski.imsave(fname, data)


# test code
img_path = './label.png'

data = ski.imread(img_path)
write2tif('mask.tif',label2binary(data))
