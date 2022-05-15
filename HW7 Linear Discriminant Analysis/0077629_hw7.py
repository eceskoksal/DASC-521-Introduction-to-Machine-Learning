#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.spatial import distance as dt
from scipy import stats

dataset=np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
y=np.genfromtxt("hw07_data_set_labels.csv", delimiter = ",")
y=y.astype(int)
K=np.max(y) #number of classes
D=dataset.shape[1]


# In[2]:



x_train=np.array(dataset[0:2000])
x_test=np.array(dataset[2000:])
y_train=np.array(y[0:2000])
y_test=np.array(y[2000:])

N_train=x_train.shape[0]
N_test=x_test.shape[0]

Mc = np.array([np.mean(x_train[y_train ==c+1],axis=0) for c in range(K)])
M= np.array([np.mean(x_train,axis=0) ])


# In[3]:


SW=np.zeros((D,D))
SB=np.zeros((D,D))

for c in range(K):
    where = np.where(y_train==c+1)[0]
    for j in where:
        SW += np.matmul( x_train[j].reshape(D,1)-Mc[c].reshape(D,1),
                        np.transpose( x_train[j].reshape(D,1)-Mc[c].reshape(D,1)))
        SB += np.matmul( Mc[c].reshape(D,1)-M.reshape(D,1),np.transpose(Mc[c].reshape(D,1)-M.reshape(D,1)))

print(SW[0:5,0:5])

print(SB[0:5,0:5])


# In[4]:


values, vectors = linalg.eig(np.matmul(np.linalg.inv(SW),SB))
values = np.real(values)
vectors = np.real(vectors)
print(values[0:9])


# In[5]:


Z_train = np.matmul(x_train - M, vectors[:,[0, 1]])
Z_test=np.matmul(x_test - M, vectors[:,[0, 1]])
# plot two-dimensional projections
plt.figure(figsize = (20, 10))


point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", 
                         "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
plt.subplot(1, 2, 1)
for c in range(K):
    plt.plot(Z_train[y_train == c + 1, 0], Z_train[y_train == c + 1, 1], 
             marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])

plt.legend(['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 
            'shirt', 'sneaker','bag', 'ankle boot'],
           loc = "upper left", markerscale = 2)
plt.xlim(right=6)
plt.xlim(left=-6)
plt.ylim(top=6)
plt.ylim(bottom=-6)
plt.xlabel("Component#1")
plt.ylabel("Component#2")

plt.subplot(1, 2, 2)
for c in range(K):
    plt.plot(Z_test[y_test == c + 1, 0], Z_test[y_test == c + 1, 1],
             marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])

plt.legend(['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
            'shirt', 'sneaker','bag', 'ankle boot'],
           loc = "upper left", markerscale = 2)
plt.xlim(right=6)
plt.xlim(left=-6)
plt.ylim(top=6)
plt.ylim(bottom=-6)
plt.xlabel("Component#1")
plt.ylabel("Component#2")

plt.show()


# In[6]:


Z_train = np.matmul(x_train - M, vectors[:,0:9])
Z_test= np.matmul(x_test - M, vectors[:,0:9])

k=11

y_predicted_train= []
distances = np.zeros(N_train)
for i in range(N_train):
    for j in range(N_train):
        distances[j]= dt.cdist(np.array([Z_train[i,:]]), np.array([Z_train[j,:]]),'euclidean')
    indices = np.argsort(distances)[:k]
    labels = []
    for x in indices:
        labels.append(y_train[x])
    prediction= stats.mode(labels)[0]
    y_predicted_train.append(prediction)
      

confusion_matrix_train = pd.crosstab(np.reshape(y_predicted_train, N_train), y_train,
                               rownames = ["y_hat"], colnames = ["y_train"])


y_predicted_test =[]
distances = np.zeros(N_train)
for i in range(N_test):
    for j in range(N_train):
        distances[j]= dt.cdist(np.array([Z_test[i,:]]), np.array([Z_train[j,:]]),'euclidean')
    indices = np.argsort(distances)[:k]
    labels = []
    for x in indices:
        labels.append(y_train[x])
    prediction= stats.mode(labels)[0]
    y_predicted_test.append(prediction)
    
confusion_matrix_test = pd.crosstab(np.reshape(y_predicted_test, N_test), y_test,
                               rownames = ["y_hat"], colnames = ["y_test"])


print(confusion_matrix_train)  
print(confusion_matrix_test)  


