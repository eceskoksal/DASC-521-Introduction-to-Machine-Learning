#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[2]:


train=np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
test=np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")


x_train=train[:,0] 
y_train=train[:,1] 
x_test=test[:,0] 
y_test=test[:,1] 



# In[3]:


bin_width = 0.1
minimum_value=0
maximum_value=2.0
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)

g=np.array([np.sum((y_train[((left_borders[i] < x_train) & (x_train <= right_borders[i]))]))  
    /np.sum((left_borders[i] < x_train) & (x_train <= right_borders[i])) for i in range(len(left_borders))])

    
plt.figure(figsize = (10, 6))   

plt.plot(x_train, y_train,"b.", markersize = 10,label="training")
plt.legend(loc='upper right')
plt.ylim(bottom=-1)
plt.ylim(top=2)
    
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g[b], g[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g[b], g[b + 1]], "k-") 
    
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.show()

plt.figure(figsize = (10, 6))   

plt.plot(x_test, y_test,"r.", markersize = 10,label="test")
plt.ylim(bottom=-1)
plt.ylim(top=2)
    
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g[b], g[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g[b], g[b + 1]], "k-") 

plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(loc='upper right')
plt.show()


# In[4]:


mse=0
for i in range(len(left_borders)): 
    for j in range(x_test.shape[0]): 
        if (left_borders[i]< x_test[j]) &  (x_test[j]<= right_borders[i] ): 
            mse=mse+(y_test[j]-g[i])**2
rmse=np.sqrt(mse/x_test.shape[0])

print(f"Regressogram => RMSE is {rmse}  when h is {bin_width}")


# In[5]:


bin_width=0.1 

data_interval = np.linspace(minimum_value, maximum_value, 1601)

g_rms = np.zeros(len(data_interval))


g_rms=np.array([np.sum(y_train[np.abs((x-x_train)/bin_width)<= 1/2])/np.sum(np.abs((x-x_train)/bin_width)<=1/2) for x in data_interval ])

plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train,"b.", markersize = 10,label="training")  
plt.plot(data_interval, g_rms, color = "k")
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize = (10, 6))   

plt.plot(x_test, y_test,"r.", markersize = 10,label="test")
plt.plot(data_interval, g_rms, color = "k")
plt.ylim(bottom=-1)
plt.ylim(top=2)
    


plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(loc='upper right')
plt.show()


# In[6]:


mse=0
for i in range(len(data_interval)-1): 
    for j in range(x_test.shape[0]): 
        if (data_interval[i]< x_test[j]) &  (x_test[j]<= data_interval[i+1] ): 
            mse=mse+(y_test[j]-g_rms[i])**2
rmse_rms=np.sqrt(mse/x_test.shape[0])

print(f"Running Mean Smoother => RMSE is {rmse_rms}  when h is {bin_width}")


# In[7]:


bin_width=0.02 
g_kernel = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)*y_train) 
                 / np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) 
                    for x in data_interval])

plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10,label="training")
plt.legend(loc='upper right')
plt.plot(data_interval,g_kernel,color="k")

plt.ylim(bottom=-1)
plt.ylim(top=2)
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")

plt.show()


plt.figure(figsize = (10, 6))
plt.plot(x_test, y_test, "r.", markersize = 10,label="test")
plt.plot(data_interval, g_kernel,color="k")

plt.ylim(bottom=-1)
plt.ylim(top=2)
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(loc='upper right')
plt.show()


# In[8]:


mse=0

for i in range(len(data_interval)-1): 
    for j in range(x_test.shape[0]): 
        if (data_interval[i]< x_test[j]) &  (x_test[j]<= data_interval[i+1] ): 
            mse=mse+(y_test[j]-g_kernel[i])**2
rmse_kernel=np.sqrt(mse/x_test.shape[0])

print(f"Kernel Smoother => RMSE is {rmse_kernel}  when h is {bin_width}")


