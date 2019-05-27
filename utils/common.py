import torch
import random
import numpy as np
from collections import OrderedDict

colors = ['#0F1F90', '#DF6767', '#67DF67','#DFA367', '#6780DF', '#8C6132', '#32798C']

def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Important also

def lprint(ms, log=None):
    '''Print message to console and to a log file'''
    print(ms)
    if log:
        log.write(ms+'\n')
        log.flush()

def plot_3d_scatter(vec, label=None, figsize=(9, 4)):
    '''Plot a vector data as a 3D scatter'''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vec[:, 0], vec[:, 1], vec[:, 2],s=1, marker=">", c='#125D4C', label=label)
    plt.show()    
    

def plot_3d_scatters(data_dict, figsize=(10, 6)):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for i, dataset in enumerate(data_dict):
        if i < len(colors):
            color = colors[i]
        else:
            color = np.random.rand(3,)
        vecs = data_dict[dataset]
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], c=color, label=dataset)
    ax.legend()
    plt.show()
    