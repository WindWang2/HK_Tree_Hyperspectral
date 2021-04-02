import glob
import os

import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as ski
import skimage.segmentation as sks
import sklearn.cluster as skc
import sklearn.mixture as skm

bands = np.load('./bands.npy')
colors = np.array([
    (255, 0, 0), # Red
    (143, 188, 143), #DarkSeaGreen
    (255, 105, 180), # Pink
    (147, 112, 219), # Purple
    (0, 255, 0), # Green
    (85, 107, 47), # OliveGreen
    (0, 0, 255), # Blue
    (139, 0, 139), # Magenta
    (184, 134, 11), # GoldenRod
    (178, 34, 34), # FireBrick
    (0, 255, 255), # Aqua
    (165, 42, 42), # Brown
    (255, 228, 196), # Bisque
])

# for step 2
def label2binary(data):
    re = np.zeros(data.shape[:-1])
    re[np.all(data == [128, 0, 0, 255], axis=-1)] = 1
    return re
# for step 2
def writeMask2tif(fname, data):
    data = np.uint8(data)
    ski.imsave(fname, data)

# for step 3
def clsByPixel(cls_data, mask_data, shape=(512, 512)):
    results_list = []
    print('Begin the kmeans: ')
    # K-mean++
    label = 'kmeans'
    y_pred = skc.KMeans(n_clusters=6).fit_predict(cls_data)
    # 255 is null data
    re_kmeans = np.ones(shape, dtype=np.uint8) * 255
    re_kmeans[mask_data == 1] = y_pred
    results_list.append((label, re_kmeans))

    # print('Begin the gaussianMix: ')
    # label = 'gaussianMix'
    # y_pred = skm.GaussianMixture(n_components=10).fit_predict(cls_data)
    # # 255 is null data
    # re_gm = np.ones(shape, dtype=np.uint8) * 255
    # re_gm[mask_data == 1] = y_pred
    # results_list.append((label, re_gm))
    return results_list

def clsBySegments(spectral_data, mask_data, rgb_path, shape=(512, 512)):
    results_list = []
    rgb_data = ski.imread(rgb_path)
    segments = sks.slic(rgb_data, n_segments=10000)+1
    segments = segments * mask_data
    unique_segments = np.unique(segments)
    # Deal with the mask with 0
    cls_data_segs = np.zeros((len(unique_segments)-1, 204))
    j = 0
    for i in unique_segments:
        if i != 0:
            cls_data_segs[j, :] = np.average(spectral_data[segments == i, :], axis=0)
            j += 1

    # K-mean ++
    label = 'kmeans_segs'
    print('Begin the {}'.format(label))
    y_pred = skc.KMeans(n_clusters=6).fit_predict(cls_data_segs)
    re_cls = np.ones(shape, dtype=np.uint8) * 255
    for i in range(len(y_pred)):
        re_cls[segments == unique_segments[i+1]] = y_pred[i]
    results_list.append((label, re_cls))

    # gaussianMix
    label = 'gaussianMix_seg'
    print('Begin the {}'.format(label))
    y_pred = skm.GaussianMixture(n_components=10).fit_predict(cls_data_segs)
    re_cls = np.ones(shape, dtype=np.uint8) * 255
    for i in range(len(y_pred)):
        re_cls[segments == unique_segments[i+1]] = y_pred[i]
    results_list.append((label, re_cls))

    # spectral clustering
    label = 'spectral_clustering_seg'
    print('Begin the {}'.format(label))
    y_pred = skc.SpectralClustering(n_clusters=10).fit_predict(cls_data_segs)
    re_cls = np.ones(shape, dtype=np.uint8) * 255
    for i in range(len(y_pred)):
        re_cls[segments == unique_segments[i+1]] = y_pred[i]

    results_list.append((label, re_cls))

    # agglometrative clustering
    label = 'agglometrative_clustering_seg'
    print('Begin the {}'.format(label))
    y_pred = skc.AgglomerativeClustering(n_clusters=10).fit_predict(cls_data_segs)
    re_cls = np.ones(shape, dtype=np.uint8) * 255
    for i in range(len(y_pred)):
        re_cls[segments == unique_segments[i+1]] = y_pred[i]
    results_list.append((label, re_cls))

    return results_list


def getMeanSpectral(spectral_data, cls_result, save_name):
    # TODO: adative for the different number of clusters
    results = np.zeros((6, 204))
    for i in range(6):
        results[i, :] = np.average(spectral_data[cls_result == i], axis=0)
    # save the spectral_data to file
    np.savetxt(save_name, results, delimiter='\t', fmt='%.4f')
    return results

def plotMeanSpectral(mean_spectral, save_name):
    # plot for each band
    for i in range(6):
        plt.plot(bands, mean_spectral[i, :], color=(0, 0, 1), linewidth=3)
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('Reflectance')
        plt.title('cluster {}'.format(i))
        plt.xlim(380, 1020)
        plt.ylim(0.00, 1)
        plt.savefig(save_name+'_{}'.format(i)+'.png')
        plt.close()
    # plot for all
    for i in range(6):
        label_str = 'cluster {}'.format(i)
        plt.plot(bands, mean_spectral[i, :],
                 color=colors[i, :] / 255.,
                 linewidth=2, label=label_str)

    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Reflectance')
    plt.title('All clusters')
    plt.xlim(380, 1020)
    plt.ylim(0.0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(save_name+'_all.png', bbox_inches="tight", dpi=300)
    plt.close()

def getMeanSpectral_with_mean(spectral_data, cls_result, clusters):
    # TODO: adative for the different number of clusters
    results = np.zeros((7, 204))
    total = 0
    for i in range(6):
        results[i, :] = np.average(spectral_data[cls_result == i], axis=0)
    for i in clusters:
        temp = spectral_data[cls_result == i]
        total += temp.shape[0]
        results[6, :] += np.sum(spectral_data[cls_result == i], axis=0)
    results[6, :] /= total
    # save the spectral_data to file
    # np.savetxt(save_name, results, delimiter='\t', fmt='%.4f')
    return results

def plotMeanSpectral_with_mean(mean_spectral, save_name, cluster_names=None):
    for i in range(6):
        if cluster_names is None:
            label_str = 'cluster {}'.format(i)
        else:
            label_str = cluster_names[i]
        plt.plot(bands, mean_spectral[i, :],
                 color=colors[i, :] / 255.,
                 linewidth=2, label=label_str)

    # plot mean
    plt.plot(bands, mean_spectral[6, :],
             color=(0, 0, 0),
             linewidth=2, label='Mean')
    plt.grid()

    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Reflectance')
    plt.title('All clusters')
    plt.xlim(380, 1020)
    plt.ylim(0.0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(save_name+'_all_with_Mean.png', bbox_inches="tight", dpi=300)
    plt.close()

def workflowByFile(input_json, input_workdir, input_spectral_file):
    # 1. step one, json 2 segments [png] in workdir

    os.system('labelme_json_to_dataset ' + '"' + input_json + '"' + ' -o ' + '"' + input_workdir + '"')

    # TODO: Check the file whether generated or not, if not raise the error
    label_file_path = os.path.join(input_workdir, 'label.png')
    mask_file_path = os.path.join(input_workdir, 'mask.tif')

    # RGB_Png file for segmentation
    # rgb_file_path = input_json.replace('json', 'png')

    # 2. step two, convert the segments [png] to mask [tif] with the value(0, 1)

    label_data = ski.imread(label_file_path)
    mask_data = label2binary(label_data)
    writeMask2tif(mask_file_path, mask_data)

    # 3. step three, classification by pixel with mask data
    spectral_data = gdal.Open(input_spectral_file).ReadAsArray().transpose((1, 2, 0))
    cls_data = spectral_data[mask_data == 1, :]
    pix_results = clsByPixel(cls_data, mask_data)
    # seg_results = clsBySegments(spectral_data, mask_data, rgb_file_path)
    # pix_results.extend(seg_results)
    # deal each results of classification
    for cc in pix_results:
        # mkdir for this classification
        cls_dir = os.path.join(input_workdir, cc[0])
        if not os.path.exists(cls_dir):
            os.mkdir(cls_dir)
        cls_file_path = os.path.join(cls_dir, 'cls.tif')
        # add color table
        cmap = np.zeros((3, 256), dtype=np.uint16)
        for i in range(colors.shape[0]):
            cmap[:, i] = np.uint16(colors[i, :])
        cmap = cmap * 256
        ski.imsave(cls_file_path, np.uint8(cc[1]), colormap=cmap)

        # plot
        table_name = os.path.join(cls_dir, 'mean_average.txt')
        mean_spectral_list = getMeanSpectral(spectral_data, cc[1], table_name)
        plot_name = os.path.join(cls_dir, 'plot_')
        plotMeanSpectral(mean_spectral_list, plot_name)

    # 4. step four, classification based on segments with mask data


def postworkperFile(input_workdir, input_spectral_file, cluster_names):

    img_file_path = os.path.join(input_workdir, 'img.png')
    mask_file_path = os.path.join(input_workdir, 'mask.tif')

    # RGB_Png file for segmentation
    # rgb_file_path = input_json.replace('json', 'png')

    # 2. step two, convert the segments [png] to mask [tif] with the value(0, 1)

    # label_data = ski.imread(label_file_path)
    # mask_data = label2binary(label_data)
    # writeMask2tif(mask_file_path, mask_data)
    mask_data = ski.imread(mask_file_path)
    img_data = ski.imread(img_file_path)
    out_img = np.zeros_like(img_data)
    spectral_data = gdal.Open(input_spectral_file).ReadAsArray().transpose((1, 2, 0))
    for i in range(3):
        out_img[:, :, i] = img_data[:, :, i] * mask_data

    out_img = ski.imsave(os.path.join(input_workdir, 'img_clip.png'), out_img)

    cls_file_dir = os.path.join(input_workdir, 'kmeans')
    cls_file_path = os.path.join(cls_file_dir, 'cls.tif')
    cc = gdal.Open(cls_file_path).ReadAsArray()
    mean_list = []
    for i, name in enumerate(cluster_names):
        if (not 'background'.upper() in name.upper()) and (not 'Saturate'.upper() in name.upper()):
            mean_list.append(i)

    mean_spectral_list = getMeanSpectral_with_mean(spectral_data, cc, mean_list)
    plot_name = os.path.join(cls_file_dir, 'plot_')
    plotMeanSpectral_with_mean(mean_spectral_list, plot_name, cluster_names=cluster_names)


if __name__ == '__main__before':
    # input_json = './testtest.json'
    # input_workdir = './new'
    # input_spectral_file = './REFLECTANCE_2019-02-25_016.dat'
    json_file_list = glob.glob('./R6_data/redo/**/*.json', recursive=True)
    json_list_name = [os.path.splitext(os.path.basename(i))[0].replace('_viz', '') for i in json_file_list]
    all_reflectance_file_list = glob.glob('../../../spectral_data/**/results/*.tif', recursive=True)
    reflectance_file_list = []
    for i in range(len(json_list_name)):
        for j in all_reflectance_file_list:
            if json_list_name[i] in j:
                reflectance_file_list.append(j)
                break

    # if not equal, raise the error
    print(len(json_file_list))
    print(len(reflectance_file_list))
    assert len(json_file_list) == len(reflectance_file_list)
    input_workdir_list = [i.replace('_viz.json', '') for i in json_file_list]
    for i, j, k in zip(json_file_list, input_workdir_list, reflectance_file_list):
        if os.path.exists(j):
            continue
        print('Begin the file {}'.format(i))
        workflowByFile(i, j, k)

# Post classification
if __name__ == '__main__':
    table = pd.read_excel('./Classes_names_Hyperspectral 1_val.xlsx')
    all_reflectance_file_list = glob.glob('/Volumes/data/polyu_work/spectral_data/**/results/*.tif', recursive=True)
    print(len(all_reflectance_file_list))
    for index, row in table.iterrows():
        # modify here
        # if row['Round'] == 6 or row['Round'] == 5:
        #     continue
        if row['Round'] != 5:
            continue
        file_name = row['Image Name ']
        cluster_list = [row['Cluster 0'], row['Cluster 1'],
                        row['Cluster 2'], row['Cluster 3'],
                        row['Cluster 4'], row['Cluster 5']]
        dir_name = os.path.join('./R5_data/val/**', file_name.replace('_all.jpg', ''))
        print(dir_name)
        dir_name = glob.glob(dir_name, recursive=True)[0]
        ref_file = ''
        for i in range(len(all_reflectance_file_list)):
            if file_name.replace('_all.jpg', '') in all_reflectance_file_list[i]:
                ref_file = all_reflectance_file_list[i]
                break
        postworkperFile(dir_name, ref_file, cluster_list)
