#!/usr/bin/env python
# coding: utf-8

# In[499]:


import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import scipy.linalg as linalg 
import pandas as pd


# In[500]:


#Question 2

np.random.seed(421)
# mean parameters
class_means = np.array([[0, +4.5],
                        [-4.5, -1],
                       [+4.5,-1],
                       [0,-4]])


# covariance parameters
class_covariances = np.array([[[+3.2, 0], 
                               [0, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]],
                             [[+1.2,-0.8],
                             [-0.8,+1.2]],
                             [[+1.2,0],
                             [0,+3.2]]])
# sample sizes
class_sizes=np.array([105,145,135,115])

points1 = np.random.multivariate_normal(mean = class_means[0,:],
                                        cov = class_covariances[0,:,:],
                                        size = class_sizes[0])
points2 = np.random.multivariate_normal(mean = class_means[1,:],
                                        cov = class_covariances[1,:,:],
                                        size = class_sizes[1])

points3 = np.random.multivariate_normal(mean = class_means[2,:],
                                        cov = class_covariances[2,:,:],
                                        size = class_sizes[2])


points4 = np.random.multivariate_normal(mean = class_means[3,:],
                                        cov = class_covariances[3,:,:],
                                        size = class_sizes[3])

x = np.vstack((points1, points2,points3,points4))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]),np.repeat(3,class_sizes[2]),
                   np.repeat(4,class_sizes[3])))


data_set=np.stack((x[:,0],x[:,1], y), axis = 1)
np.savetxt("hw1_dataset.csv", (np.stack((x[:,0],x[:,1],y), axis = 1)), fmt = "%f,%f,%d")

# read data into memory
data_set = np.genfromtxt("hw1_dataset.csv", delimiter = ",")

# get x and y values
x = data_set[:,(0,1)]
y = data_set[:,2].astype(int)


# get number of classes and number of samples
K = np.max(y)  #number of classes
N = data_set.shape[0] #nubmer of data points


plt.figure(figsize = (8, 8))
plt.plot(points1[:,0], points1[:,1],"r.", markersize = 20)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 20)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 20)
plt.plot(points4[:,0], points4[:,1], "m.", markersize = 20)  

plt.plot()
plt.xlim((-8, +8))
plt.ylim((-8, +8))
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()


# In[501]:


# Question 3

sample_means = np.array([np.mean(x[y == (c + 1)],axis=0) for c in range(K)])
print(sample_means)


sample_covariances=np.array([(np.matmul(np.transpose(x[y == (c + 1)]-sample_means[c]),
                                        x[y == (c + 1)]-sample_means[c]))/class_sizes[c] for c in range(K)])
print(sample_covariances)


class_priors = np.array([np.mean(y == (c + 1)) for c in range(K)])
print(class_priors)






# In[502]:


#Question 4

W=np.array([-1/2*(np.linalg.inv(sample_covariances[c])) for c in range(K)])
w=np.array([np.matmul(np.linalg.inv(sample_covariances[c]),sample_means[c]) for c in range(K)])
w0=np.array([-1/2*(np.matmul(np.transpose(sample_means[c]),
                         np.matmul(np.linalg.inv(sample_covariances[c]),
                                   sample_means[c])))-
              1/2*class_means.shape[1]*np.log(math.pi) -
             1/2*np.log(np.linalg.det(sample_covariances[c])) +
              np.log(class_priors[c]) for c in range(K)])  

def g(x):
    g=np.array([0,0,0,0])
    for c in range(K):
        g[c]= np.matmul(np.matmul(np.transpose(x), W[c]), x)+np.matmul(np.transpose(w[c]), x)+w0[c]
    return g

y_truth=y 
y_pred=np.argmax([g(x[i]) for i in range(N)],axis=1)+1


confusion_matrix = pd.crosstab(y_pred, y_truth, 
                               rownames = ["y_pred"], 
                               colnames = ["y_truth"])

print(confusion_matrix)


# In[503]:


x1_interval = np.linspace(-8, +8, 1001)
x2_interval = np.linspace(-8, +8, 1001)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((x1_interval.shape[0],x2_interval.shape[0], K))


for i in range(K):
    for j in range(x1_grid.shape[0]):
        for k in range(x2_grid.shape[0]):
            D = np.array([x1_grid[j][k], x2_grid[j][k]]).reshape(2, 1)
            discriminant_values[j, k, i] = np.matmul(np.matmul(np.transpose(D), W[i]), D) + np.matmul(np.transpose(w[i]), D) + w0[i]


# In[512]:


#Question 5
A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
D= discriminant_values[:, :, 3]

A[(A<B)&(A<C)]=np.nan
B[(B<D)&(B<C)]=np.nan
C[(C<A)&(C<B)]=np.nan
D[(D<A)&(D<B)]=np.nan

discriminant_values[:, :, 0]=A
discriminant_values[:, :, 1]=B
discriminant_values[:, :, 2]=C
discriminant_values[:, :, 3]=D

plt.figure(figsize = (8, 8))
plt.plot(x[y_truth == 1, 0], x[y_truth == 1, 1], "r.", markersize = 15)
plt.plot(x[y_truth == 2, 0], x[y_truth == 2, 1], "g.", markersize = 15)
plt.plot(x[y_truth == 3, 0], x[y_truth == 3, 1], "b.", markersize = 15)
plt.plot(x[y_truth == 4, 0], x[y_truth == 4, 1], "m.", markersize = 15)

plt.plot(x[y_pred != y_truth, 0], x[y_pred != y_truth, 1],"ko", markersize = 18, fillstyle = "none")

plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,3], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,2] - discriminant_values[:,:,3], levels=0, colors="k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,3], levels=0, colors="k")



plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()


# In[ ]:




