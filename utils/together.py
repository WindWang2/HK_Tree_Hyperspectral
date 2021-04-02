import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color as skc
import skimage.io as ski
import gdal
if __name__== '__main__':
    workdir = './R5_data/val/'
    table = pd.read_excel('./Classes_names_Hyperspectral 1_val.xlsx')
    json_file_list = glob.glob(os.path.join(workdir, '**/*.json'), recursive=True)
    json_list_name = [os.path.splitext(os.path.basename(i))[0].replace('_viz', '') for i in json_file_list]
    print(json_list_name)
    for i, j in zip(json_list_name, json_file_list):
        workdir = os.path.dirname(j)
        fig = plt.figure()
        axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

        print(i)

        # rgb
        rgb = os.path.join(workdir, i, 'img.png')
        rgb_data = np.rot90(ski.imread(rgb), k=3, axes=(0, 1))
        axes[0].imshow(rgb_data)


        # label
        label = os.path.join(workdir, i, 'kmeans', 'cls.tif')
        label_data = np.rot90(plt.imread(label), k=3, axes=(0, 1))
        axes[1].imshow(label_data)

        # plot

        plot = os.path.join(workdir, i, 'kmeans', 'plot__all_with_Mean.png')
        if not os.path.exists(plot):
            continue

        out_row = table[table['Image Name '] == i]
        ID = out_row['Tree ID'].values[0]
        species = out_row['Species'].values[0]
        plot_data = gdal.Open(plot).ReadAsArray().transpose((1, 2, 0))
        axes[2].imshow(plot_data)

        # combine
        gray_data = skc.rgb2gray(rgb_data)
        re = np.zeros((512, 512, 3))
        rgb_data[np.all(label_data == np.array([0, 0, 0, 255]), axis=-1)] = np.array([0, 0, 0])
        axes[3].imshow(rgb_data)
        # axes[3].imshow(label_data, alpha=0.4)
        fig.suptitle('{}({})'.format(species, ID))
        for ax in axes:
            ax.axis('off')
        # fig.tight_layout()
        # fig.subplots_adjust(hspace=0.11, wspace=0)
        fig.savefig(os.path.join(workdir, i, i+'_all.jpg'), dpi=600)
        plt.close()
        # break
