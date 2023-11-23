#!/usr/bin/python3
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=X.shape)

X_noisy = X + noise

SNR = 10 * np.log10((np.linalg.norm(X) ** 2) / (np.linalg.norm(noise) ** 2))

plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()

plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title(f'Noisy with SNR = {SNR}')
plt.savefig('NoisyRat.png')
plt.show()

Y = np.fft.fft2(X)
freq_db = 20 * np.log10(np.abs(Y))

Y_cutoff = np.copy(Y)

freq_cutoff = 250

Y_cutoff[freq_db > freq_cutoff] = 0

X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)  # avoid rounding erros in the complex domain,
# in practice use irfft2

noise = X - X_cutoff

SNR = 10 * np.log10((np.linalg.norm(X) ** 2) / (np.linalg.norm(noise) ** 2))

plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.title(f'Denoised with SNR = {SNR}')
plt.savefig('DenoisedRat.png')
plt.show()
