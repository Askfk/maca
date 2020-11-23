from PIL import Image
import matplotlib.pyplot as plt
import csv

import numpy as np


def plotImage(root_dir, name):
    image_path = root_dir + name + '.jpeg'
    gaze_path = root_dir + name + '.csv'

    image = Image.open(image_path)
    file = csv.reader(open(gaze_path, 'r'))

    image = image.convert("RGB")
    _, ax = plt.subplots(1)

    data = [list(map(float, x)) for x in file]
    data = np.array(data)
    print(data.shape)
    ix = np.where(data >= 0)
    data = data[ix]
    data = np.reshape(data, [data.shape[0] // 2, 2])

    ax.imshow(image)
    plt.scatter(data[:, 0], data[:, 1], c='#00CED1', alpha=0.4)
    plt.show()


if __name__ == '__main__':
    name = 'i114742201_lowRes512'
    root_dir = '/Users/liyiming/Desktop/gaze/data/picked data/'
    plotImage(root_dir, name)
