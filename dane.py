import hickle as hkl
import numpy as  np
import matplotlib.pyplot as plt
filename = 'haberman.txt'

data = np.loadtxt(filename,delimiter = ',',dtype= str)

x = data[:,0:-1].astype(float).T
y_t = data[:,-1].astype(float)
y_t = y_t.reshape(1,y_t.shape[0])
x_min = x.min(axis=1)
x_max = x.max(axis=1)
x_norm_max = 1
x_norm_min = -1
x_norm = np.zeros(x.shape)
for i in range(x.shape[0]):
 x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])*(x[i,:]-x_min[i]) + x_norm_min

y_t_s_ind = np.argsort(y_t)
x_n_s = np.zeros(x.shape)
y_t_s = np.zeros(y_t.shape)
for i in range(x.shape[1]):
 y_t_s[0,i] = y_t[0,y_t_s_ind[0,i]]
 x_n_s[:,i] = x_norm[:,y_t_s_ind[0,i]]
plt.plot(y_t_s[0])

hkl.dump([x,y_t,x_norm,x_n_s,y_t_s],'haberman.hkl')

