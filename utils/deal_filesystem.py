import os

import pandas as pd

all_file_list = open('./fsystem_v3.txt').readlines()

# read the excel

tables = pd.read_excel('./Classes_names_Hyperspectral 1.xlsx')
image_name_pd = tables.loc[:, ['Image Name ']]
image_name_list = [i[0].replace('_all.jpg', '') for i in image_name_pd.values.tolist()]
print(len(image_name_list), len(all_file_list))
out = []

for i in image_name_list:
    for j in all_file_list:
        if i in j:
            out.append(j)
            break

print(len(out))
fs = open('./fsystem_sub_v3.txt', 'w')
fs.writelines(out)
fs.close()
