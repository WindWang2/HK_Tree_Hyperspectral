import sys

import gdal
import numpy as np
import skimage.io as ski

bands = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
infile_path = sys.argv[1]
outfile_path = sys.argv[2]

def spectral2png(in_path, out_path, band_vector):
    data = gdal.Open(in_path).ReadAsArray()
    # from (band, x, y) to (x, y, band)
    data = data.transpose((1,2,0))
    data = data[:, :, band_vector] * 300
    data[data > 255] = 255
    outdata = np.uint8(data)
    ski.imsave(out_path, outdata)

if __name__ == '__main__':
    spectral2png(infile_path, outfile_path, bands)
