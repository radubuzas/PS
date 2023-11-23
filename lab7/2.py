#!/usr/bin/python3
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)

Y = np.fft.fft2(X)
freq_db = 20 * np.log10(np.abs(Y))

SNR = 0

freq_cutoff = 10

while SNR < 100:
    Y_cutoff = np.copy(Y)
    Y_cutoff[freq_db > freq_cutoff] = 0

    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)

    noise = X - X_cutoff
    SNR = 10 * np.log10(np.linalg.norm(X) ** 2 / (np.linalg.norm(noise) ** 2))

    plt.title(f'SNR: {SNR}\nEliminated frequencies above {freq_cutoff} Hz')
    plt.imshow(X_cutoff, cmap=plt.cm.gray)
    plt.show()

    freq_cutoff += 10
