import glob
import os

import gdal
# import matplotlib.pyplot as plt
import numpy as np
import skimage.io as ski

# read data from spectral cube

def get_viz_data(rgb_raw):
    """
    Visualization by linear 2%
    Keyword Arguments:
    rgb_raw -- raw data of spectral reflectance
    """
    # Stat
    for i in range(3):
        temp_data = rgb_raw[:, :, i]
        stat, edge = np.histogram(temp_data, bins=300)
        cum_stat = np.cumsum(stat)
        # %2 and %98 for low and up bound, respectively
        low = edge[0:-1][cum_stat > (512*512*0.02)][0]
        up = edge[1:][cum_stat > (512*512*0.98)][0]

        temp_data[temp_data < low] = low
        temp_data[temp_data > up] = up

        temp_data = (temp_data-low) / (up-low)
        rgb_raw[:, :, i] = temp_data

def get_viz_file(in_file, out_file):
    spectral_data = gdal.Open(in_file).ReadAsArray()
    rgb_data = spectral_data[(69, 52, 18), :, :]
    rgb_data = rgb_data.transpose((1, 2, 0))
    get_viz_data(rgb_data)

    ski.imsave(out_file, np.uint8(rgb_data * 255))

if __name__ == '__main__':
    # spectral_data = gdal.Open('./REFLECTANCE_2019-02-25_016.dat').ReadAsArray()
    # rgb_data = spectral_data[(69, 52, 18), :, :]
    # rgb_data = rgb_data.transpose((1,2,0))
    # ski.imsave('./tet.png',np.uint8(rgb_raw*255))
    input_dir = '/Volumes/BOOTCAMP/Users/kevin/Desktop/'
    search_reg = os.path.join(input_dir, '**/results/*.dat')
    spec_file_list = glob.glob(search_reg, recursive=True)
    print("Rotal numboer of spectral files is {:d}".format(len(spec_file_list)))
    for i in spec_file_list:
        print(i)
        out_file = os.path.join('./viz', os.path.splitext(os.path.basename(i))[0]+'_viz.png')
        get_viz_file(i, out_file)
    # get_viz_file('./REFLECTANCE_2019-02-25_016.dat', './testtest.png')
    # plt.show()
