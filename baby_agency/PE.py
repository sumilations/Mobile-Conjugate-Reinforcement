import numpy as np
import pandas as pd
from entropy import *
import matplotlib.pyplot as plt

series = np.loadtxt("baby_filtered.csv")
X = series[:]
X = (X-np.min(X))/float(np.max(X) - np.min(X))

print(len(X))

X = X.flatten()
X = list(X)
r = np.linspace(0,60,len(X)) #series[:,0]

slide = 1
size = 512
overlap = 511
series_len = len(X)
entropy = []
snn_mode_a = []
snn_mode_b = []


y = [l for l in zip(*[X[i::(size-overlap)] for i in range(size)])]

print(type(list(y[200])))

for i in range(0,len(y)):
	
	entropy.append(permutation_entropy(y[i], order=4, delay=1, normalize=True))


entropy = np.array(entropy)	


print(len(entropy))
	
np.savetxt('HKB_entropy_df_256.csv', np.c_[r[0:len(y)], X[0:len(y)], entropy])

fig = plt.figure(figsize=(5,3))

plt.plot(r[0:len(y)], entropy)
plt.xlabel("$\delta\omega$")
plt.ylabel("$PE$")
plt.savefig("PE_HKB_DF.pdf")
plt.show()	
	
