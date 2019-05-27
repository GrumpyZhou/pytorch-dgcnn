
def plot_3d_scatter(vec, label=None):
    '''Plot a vector data as a 3D scatter'''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vec[:, 0], vec[:, 1], vec[:, 2],s=1, marker=">", c='#125D4C', label=label)
    plt.show()