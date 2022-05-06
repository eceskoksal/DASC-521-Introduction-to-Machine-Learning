#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


#2
x=np.genfromtxt("hw03_data_set_images.csv", delimiter = ",")
y=np.genfromtxt("hw03_data_set_labels.csv", delimiter = ",")

N=x.shape[0] 
K=np.max(y).astype(int)


training_set_dim=25*K
test_set_dim=N-training_set_dim


        


# In[3]:


#3
x_train=np.zeros((training_set_dim,x.shape[1]))
x_test=np.zeros((test_set_dim,x.shape[1]))
y_train=np.array(np.zeros((training_set_dim)))
y_test=np.array(np.zeros((test_set_dim)))

y_truth_train = np.zeros((training_set_dim, K)).astype(int)
y_truth_test = np.zeros((test_set_dim, K)).astype(int)


count1=0
count2=0
for i in range(x.shape[0]):
    if i%39 <25:  
        x_train[count1,:]=x[i,:]
        y_train[count1]=y[i]
        y_truth_train[count1,y[i].astype(int)-1]=1
        
        count1=count1+1
    
    else:
        x_test[count2,:]=x[i,:]
        y_test[count2]=y[i]
        y_truth_test[count2,y[i].astype(int)-1]=1
        
        count2=count2+1


# In[4]:


# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

def gradient_W(X, Y_truth, Y_predicted):
    return(np.asarray([-np.matmul(Y_truth[:,c] - Y_predicted[:,c], X) for c in range(K)]).transpose())


def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))



eta=0.001
epsilon=0.001

np.random.seed(521)

W = np.random.uniform(low = -0.01, high = 0.01, size = (x_train.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))

# learn W and w0 using gradient descent
iteration = 1
objective_values = []

while True:
    y_predicted = sigmoid(x_train, W, w0) 
    objective_values = np.append(objective_values, 0.5*np.sum((y_truth_train-y_predicted)**2))

    W_old = W
    w0_old = w0




    W = W - eta * gradient_W(x_train, y_truth_train, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth_train, y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1

print(W)
print(w0)


# In[5]:


# plot objective function during iterations
plt.figure(figsize = (8, 4))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[6]:


Y_predicted = np.argmax(y_predicted, axis = 1) + 1
y_train=y_train.astype(int)
confusion_matrix_train = pd.crosstab(Y_predicted, y_train, rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix_train)


# In[7]:


test_results=sigmoid(x_test,W,w0)
test_predicted=np.argmax(test_results, axis = 1) + 1
y_test=y_test.astype(int)

print(test_predicted.shape)
print(y_test.shape)
confusion_matrix_test = pd.crosstab(test_predicted, y_test, 
                              rownames = ["y_pred"], 
                              colnames = ["y_truth"])

print(confusion_matrix_test)


# In[ ]:





# In[ ]:




