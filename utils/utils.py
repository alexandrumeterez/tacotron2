import matplotlib.pyplot as plt
import numpy as np


def show_spectrogram(spec, text=None):
    plt.figure(figsize=(6, 14))
    plt.imshow(np.transpose(spec))
    if text:
        plt.title(text, fontsize='10')
    plt.ylabel('mels')
    plt.xlabel('frames')
    plt.show()


