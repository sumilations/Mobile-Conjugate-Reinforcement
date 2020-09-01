import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 3
fs = 1000       # sample rate, Hz
cutoff = 10 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.

w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
data = np.loadtxt('baby_co_vel.csv')
n = np.shape(data)[0] # total number of samples

t =  np.arange(1, n+1, 1)
y = butter_lowpass_filter(data, cutoff, fs, order)

diff = np.gradient(y)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
np.savetxt("baby_filtered.csv", np.c_[y])
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.figure(2)

plt.plot(diff)

plt.subplots_adjust(hspace=0.35)
plt.show()
