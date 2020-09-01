import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

v = np.loadtxt("snn_day_filtered.csv")
v = (v - np.min(v))/(np.max(v)-np.min(v))
v = v[1:]
cor = np.correlate(v,v,mode='full')
dt = 0.001
sampling = 1./dt
freqs, psd = signal.welch(v, sampling, nperseg =4096)
freqs = freqs[10:]
psd = psd[10:]
print(freqs,psd)

def func(x, a, b):
	return b/(x**(a))

print(len(psd))
popt, pcov = curve_fit(func, freqs, psd)#, p0=(1, 0.2))
print(popt)
plt.loglog(freqs[100:600], func(freqs[100:600], 3.14, 6.9), 'g',label='~$1/{f^{3}}$', linewidth = 3)
plt.legend()
plt.xlabel("f")
plt.ylabel("psd")
plt.loglog(freqs,psd)
plt.show()
