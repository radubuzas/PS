from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)

Y = np.fft.fft2(X)
freq_db = 20 * np.log10(np.abs(Y))

Y_cutoff = np.copy(Y)

freq_cutoff = 120

Y_cutoff[freq_db > freq_cutoff] = 0



X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2

a = np.linalg.norm(X) ** 2
b = np.linalg.norm(X_cutoff) ** 2

SNR: float = 100

print(np.sqrt(a / (b * SNR)))

# plt.title('Cutoff')
# plt.imshow(X_cutoff, cmap=plt.cm.gray)
# plt.show()