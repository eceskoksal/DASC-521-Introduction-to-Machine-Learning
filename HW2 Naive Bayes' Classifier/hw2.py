#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


#2
x=np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")
y=np.genfromtxt("hw02_data_set_labels.csv", delimiter = ",")

N=x.shape[0] 
K=np.max(y).astype(int)


training_set_dim=25*K
test_set_dim=N-training_set_dim


# In[3]:


#3
x_train=np.zeros((training_set_dim,x.shape[1]))
x_test=np.zeros((test_set_dim,x.shape[1]))
y_train=np.zeros((training_set_dim))
y_test=np.zeros((test_set_dim))

count1=0
count2=0
for i in range(x.shape[0]):
    if i%39 <25:  
        x_train[count1,:]=x[i,:]
        y_train[count1]=y[i]
        count1=count1+1
    else:
        x_test[count2,:]=x[i,:]
        y_test[count2]=y[i]
        count2=count2+1
        


# In[4]:


#4 

pcd= np.array([np.sum(x_train[y_train == (c+1)],axis=0)/np.sum(y_train==(c+1),axis=0) for c in range(K)])
print(pcd)

class_priors = np.array([np.mean(y_train == (c + 1))for c in range(K)])
print(class_priors)

    


# In[5]:


#5 
def safelog(x):
    return(np.log(x + 1e-100))

def g(x): 
    g=np.zeros(K)
    for c in range(K):
        g[c]=np.sum(x*safelog(pcd[c]) + (1-x)*safelog(1-pcd[c])) + safelog(class_priors[c])   
    return g 

 
y_truth_train=y_train.astype(int)
y_pred_train=np.zeros(x_train.shape[0]).astype(int)

for i in range(x_train.shape[0]): 
    y_pred_train[i]=np.argmax([g(x_train[i])]) +1


confusion_matrix_train = pd.crosstab(y_pred_train, y_truth_train, 
                               rownames = ["y_pred"], 
                               colnames = ["y_truth"])

print(confusion_matrix_train)


# In[6]:


#6

y_truth_test=y_test.astype(int)
y_pred_test=np.zeros(x_test.shape[0]).astype(int)

for i in range(x_test.shape[0]): 
    y_pred_test[i]=np.argmax([g(x_test[i])]) +1
    

confusion_matrix_test = pd.crosstab(y_pred_test, y_truth_test, 
                               rownames = ["y_pred"], 
                               colnames = ["y_truth"])

print(confusion_matrix_test)


# In[ ]:





# In[ ]:




