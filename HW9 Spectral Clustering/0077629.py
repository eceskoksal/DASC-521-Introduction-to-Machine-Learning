#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import distance as dt
import scipy.spatial as spa
import scipy.linalg as linalg

dataset=np.genfromtxt("hw09_data_set.csv", delimiter = ",")
N=dataset.shape[0]

plt.figure(figsize = (8, 8))
plt.plot(dataset[:,0], dataset[:,1],"k.", markersize = 10)

plt.plot()
plt.xlim((-8, +8))
plt.ylim((-8, +8))
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()


# In[2]:


B=np.zeros((N,N))

delta=2

distance_matrix= dt.cdist(dataset,dataset,'euclidean')


for i in range(N):
    for j in range(N):
        if i != j: 
            if distance_matrix[i,j]< delta  :
                B[i,j]=1
            else: 
                B[i,j]=0
        else: 
            B[i,j]=0 


plt.figure(figsize = (8, 8))

for i in range(N):
    for j in range(N): 
        if B[i,j]==1:
            x_values=dataset[i,0],dataset[j,0]
            y_values=dataset[i,1],dataset[j,1]
            plt.plot(x_values,y_values,color="gray",linestyle="-",markersize=5)
plt.plot(dataset[:,0], dataset[:,1],"k.", markersize = 10)

plt.xlim((-8, +8))
plt.ylim((-8, +8))
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()


# In[3]:


D=np.zeros((dataset.shape[0],dataset.shape[0]))

d=B

for i in range(N):
    for j in range(N): 
        if i == j: 
            d[i,j]=0


for i in range(N):
    D[i,i]=np.sum(d[i,:])

L=D-B 

L_symmetric=np.eye(N)-np.matmul(np.matmul(linalg.fractional_matrix_power(D,-0.5),B),
                                linalg.fractional_matrix_power(D,-0.5))

print(L_symmetric[0:5,0:5])


# In[4]:


values, vectors = linalg.eig(L_symmetric)
values = np.real(values)
R=5 
      
indices=np.argsort(values)

Z=np.zeros((N,R))

for r in range(R):
    Z[:,r]=vectors[:,indices[r+1]]

print(Z[0:5,0:5])



# In[5]:


initial_centroids=np.array([Z[242,:], Z[528,:], Z[570,:],Z[590,:], Z[648,:],
                            Z[667,:], Z[774,:], Z[891,:], Z[955,:]])

K=9 

def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids =initial_centroids
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
       
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
         
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
    
        plt.plot(centroids[c,0],centroids[c,1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    
    plt.xlim((-8, +8))
    plt.ylim((-8, +8))
    plt.xlabel("$x_{1}$")
    plt.ylabel("$x_{2}$")
   


    
centroids = None
memberships = None
iteration = 1

while True:
   
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break
        
    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break
    iteration = iteration + 1
    
plt.figure(figsize = (8, 8))
plot_current_state(centroids, memberships, dataset)

