from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np


def show_spectrogram(spec, text=None, return_array=False):
    plt.figure(figsize=(6, 14))
    plt.imshow(np.transpose(spec))
    if text:
        plt.title(text, fontsize='10')
    # plt.colorbar(shrink=0.5, orientation='vertical')
    plt.ylabel('mels')
    plt.xlabel('frames')
    plt.show()
