import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('./indexes.xlsx')

abbr_ls = np.load('abbr.npy')
idx_str = 'PRI'

Round_label_lis = [' R' + str(i) for i in range(6)]
month_label = ['Dec', 'Feb', 'Apr', 'Jun', 'Aug', 'Oct']
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

for idx_sp in abbr_ls:
    fig, ax = plt.subplots(figsize=(4,3))
    mean_lis = []
    for idx_rn, rn in enumerate(Round_label_lis):
        y = df.loc[(df['SpShort'] == idx_sp) & (df['Round'] == rn), idx_str].to_numpy()
        x = np.ones(len(y)) * (idx_rn + 1)
        mean_lis.append(np.mean(y))
        plt.plot(x, y, color=colors[3, :]/255., linestyle='', marker='x')

    plt.plot(range(1, len(mean_lis)+1), mean_lis, marker='*', linestyle='', color='b')

    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    plt.xlim(0.5, 6.5)
    plt.ylim(-.1, .1)
    ax.set_xticklabels([''] + month_label, fontsize=10)
    plt.xlabel('Month')
    plt.ylabel(idx_str)
    plt.title(idx_sp)
    plt.savefig('./temp/{}_{}.png'.format(idx_str, idx_sp), bbox_inches='tight', dpi=300)
