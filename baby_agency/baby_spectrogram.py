import numpy as np
from scipy import signal
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt

v = np.loadtxt("HKB.csv")
v= v[:,1]
fs = 1000

f, t, Sxx = signal.spectrogram(v, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
