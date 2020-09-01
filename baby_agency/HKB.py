import numpy as np
from random import gauss
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
#import colorednoise as cn

a = 1
b = 1

dt    = 0.01
final_time = 30
nsteps = int(final_time/dt)
dw = 0
D = 0.01
sampling = 1./dt


def f(state,t,dw):
  phi = state   # unpack the state vector  
  dphi = dw - a*np.sin(phi) - 2*b*np.sin(2.*phi)  
  return  dphi 

DW = 3.8
state0 = [0]#[-np.sqrt(beta*(rho - 1.)), -np.sqrt(beta*(rho - 1.)) , (rho - 1.)]
state_begin = state0
print(state0)
t = np.linspace(0, dt, 2)
state_buffer =[]
noise_buffer = []
vel = []
dphi_list = []
dw_list = []


for i in range(0,nsteps):
	N1 = D*np.sqrt(dt)*gauss(0, 1)
	dw_list.append(dw)
	dw = dw + dt
	states = odeint(f, state0, t, args=(4,))
	states[-1] = states[-1] + np.array([N1])
	states[-1] = states[-1]%(2*np.pi) 
	dphi =  f(states[-1],0, DW )
	state_buffer.append(states[-1])
	dphi_list.append(dphi)
	noise_buffer.append([N1])	
	state0 = states[-1]
	
t0 = np.linspace(0,nsteps*dt,nsteps)	

np.savetxt('HKB_meta.csv', np.c_[t0, state_buffer[:], dphi_list[:]])

#state_buffer = np.array(state_buffer)
noise_buffer = np.array(noise_buffer)
dphi_list = np.array(dphi_list)
dw_list = np.array(dw_list)

print(np.shape(dphi_list), np.shape(dw_list), np.shape(state_buffer))
fig = plt.figure(figsize=(5,3))

ax = fig.add_subplot(111)#, projection = '3d')

ax.plot(t0, state_buffer[:])#, linewidth=0.5, alpha = 0.8, c="blue")#, label = "controlled")
plt.xlabel("$\delta\omega$")
plt.ylabel("$\dot\phi$")

plt.savefig("HKB_meta_2.pdf", bbox_inches='tight')
plt.show()

