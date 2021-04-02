import glob
import os

import gdal


def envi2gdal(infile_path, outfile_path):
    ds = gdal.Open(infile_path)
    driver = gdal.GetDriverByName('GTiff')
    re_ds = driver.CreateCopy(outfile_path, ds, strict=0)
    re_ds = None
    ds = None
    return 0

if __name__ == '__main__':
    input_dir = './Round 6 _ Validation'
    search_reg = os.path.join(input_dir, '**/results/*.dat')
    spec_file_list = glob.glob(search_reg, recursive=True)
    for i, sfile in enumerate(spec_file_list):
        print('{}/{}'.format(i, len(spec_file_list)), sfile)
        out_sfile = sfile.replace('.dat', '.tif')
        if os.path.exists(out_sfile):
            continue
        envi2gdal(sfile, out_sfile)
