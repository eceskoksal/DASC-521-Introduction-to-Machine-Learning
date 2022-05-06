#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import cvxopt as cvx
import matplotlib.pyplot as plt

dataset=np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
labels=np.genfromtxt("hw06_data_set_labels.csv", delimiter = ",")
K=2 #number of classes
D=dataset.shape[1]



# In[2]:


x_train=np.array(dataset[0:1000])
x_test=np.array(dataset[1000:])
y_train=np.array(labels[0:1000])
y_test=np.array(labels[1000:])


# In[3]:


number_bins=64

N_train=x_train.shape[0]
N_test=x_test.shape[0]

minimum_value=0 
maximum_value=256
bin_width=maximum_value/number_bins

left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)



H_train=np.zeros((N_train,number_bins))
H_test=np.zeros((N_test,number_bins))

for i in range(N_train):
    for b in range(len(left_borders)):
        H_train[i,b] = np.sum((left_borders[b] <= x_train[i,:]) & (x_train[i,:] < right_borders[b])) / D
        
for i in range(N_test):
    for b in range(len(left_borders)):
        H_test[i,b] = np.sum((left_borders[b] <= x_test[i,:]) & (x_test[i,:] < right_borders[b])) / D
               

print(H_train[0:5, 0:5])
print(H_test[0:5, 0:5])


# In[4]:


L=number_bins 

K_train=np.zeros((N_train,N_train))
K_test=np.zeros((N_test,N_train))

sm=0

for i in range(N_train):
    for j in range(N_train): 
        for l in range(L): 
            sm = sm + np.min((H_train[i,l],H_train[j,l]))
        K_train[i,j]=sm 
        sm=0
        
for i in range(N_test):
    for j in range(N_train): 
        for l in range(L): 
            sm = sm + np.min((H_test[i,l],H_train[j,l]))
        K_test[i,j]=sm 
        sm=0
        
print(K_train[0:5,0:5])
print(K_test[0:5,0:5])


# In[5]:


def kernelmachine(C,y_train,N_train,K_train):
    s =1
    yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train

    # set learning parameters
    epsilon = 0.001

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None,:])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    return alpha,w0

alpha,w0=kernelmachine(10,y_train,N_train,K_train)

f_predicted_train = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0

f_predicted_test = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0


# calculate confusion matrix
y_train=y_train.astype(int)
y_test=y_test.astype(int)

y_predicted_train = 2 * (f_predicted_train > 0.0) - 1
y_predicted_test= 2 * (f_predicted_test > 0.0) - 1

confusion_matrix_train = pd.crosstab(np.reshape(y_predicted_train, N_train), y_train,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix_train)

confusion_matrix_test = pd.crosstab(np.reshape(y_predicted_test, N_test), y_test,
                               rownames = ["y_predicted"], colnames = ["y_test"])

print(confusion_matrix_test)



# In[6]:


accuracy_test = []
accuracy_train=[]

C_matrix=np.array([[10**(-1)],[10**(-0.5)],[10**0],[10**1],[10**1.5],[10**2],[10**2.5],[10**3]])

for c in range(len(C_matrix)):
    C=C_matrix[c]
    
    alpha,w0=kernelmachine(C,y_train,N_train,K_train)
    
    f_predicted_train=np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
    f_predicted_test=np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0
    
    y_train_pred=2 * (f_predicted_train > 0.0) - 1
    y_test_pred=2 * (f_predicted_test > 0.0) - 1
    
    sm=0
    for i in range(len(y_train)):
        if y_train[i]==y_train_pred[i]:
            sm=sm+1
    accuracy=sm/len(y_train)
    
    accuracy_train.append(accuracy)
    s=0
    for j in range(len(y_test)):
        if y_test[j]==y_test_pred[j]:
            s=s+1
            
    acc_test=s/len(y_test)
    accuracy_test.append(acc_test)          
    
fig = plt.figure(figsize=(10,5))
plt.plot(C_matrix,accuracy_train,color= "b",label="training",marker="o")
plt.plot(C_matrix,accuracy_test,color= "r",label="test",marker="o")

plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
#plt.ticklabel_format(scilimits='sci')
plt.xscale('log',base=10) 

plt.legend(loc="upper left")
plt.show()


      
    


# In[ ]:





# In[ ]:




