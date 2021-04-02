import glob
import os
import re
import shutil


def mkdir(s_dir, d_dir):
    dire = os.path.join('./test', *d_dir)
    if not os.path.exists(dire):
        os.makedirs(dire)
    shutil.copytree(os.path.join('./', s_dir), os.path.join(dire, s_dir))

    return 0

def findRe(ss):
    print(ss)
    # find tree name
    tree_name = re.findall('temp\/(.*)\/r\d\/', ss, flags=re.I)[0]

    # find time (round)
    round = re.findall('\/(r\d)\/', ss, flags=re.I)[0]

    #find ID
    id = re.findall('\_(rtr\d{7})', ss, flags=re.I)[0]
    return [tree_name, id, round]



if __name__ == '__main__':
    json_file_list = glob.glob('./*.json')
    json_list_name = [os.path.splitext(os.path.basename(i))[0].replace('_viz', '') for i in json_file_list]
    # read file system
    ls = open('./fsystem.txt').readlines()

    for i in json_list_name:
        tmp = ''
        for j in ls:
            if i in j:
                tmp = j
                break
        out_re = findRe(tmp)
        mkdir(i, out_re)
