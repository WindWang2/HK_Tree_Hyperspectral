import glob
import os
import sys

json_list = glob.glob('./spectral_common/viz_R4_clipping/*.json')
# exe
[os.system('labelme_json_to_dataset ' + i + ' -o ' + os.path.splitext(i)[0]) for i in json_list]
