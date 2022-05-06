#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 

    
    
train = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
test = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

x_train = train[:,0]
y_train= train[:,1]
x_test =test[:,0]
y_test= test[:,1]


# In[2]:


def tree(x_train,y_train,P):
  
    node_indices = {}
    is_terminal = {}
    need_split = {}
    
    node_means = {}
    node_splits = {}

 
    node_indices[1] = np.array(range(x_train.shape[0]))
    is_terminal[1] = False
    need_split[1] = True
    
    while True:
        
       
        split_nodes = [key for key, value in need_split.items() if value == True]
        if len(split_nodes) == 0:
            break
            
        
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])
        
            if x_train[data_indices].size <= P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean
                
            else:
                is_terminal[split_node] = False
                x_sorted = np.sort(np.unique(x_train[data_indices]))
                split_positions = (x_sorted[1:len(x_sorted)] +x_sorted[0:(len(x_sorted)-1)])/2
                split_scores = np.repeat(0.0,len(split_positions))
                
                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] < split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] >= split_positions[s]]
                    error = 0
                    if len(left_indices)>0:
                        error += np.sum((y_train[left_indices] - np.mean(y_train[left_indices])) ** 2)
                    if len(right_indices)>0:
                        error += np.sum((y_train[right_indices] - np.mean(y_train[right_indices])) ** 2)
                    split_scores[s] = error/(len(left_indices)+len(right_indices))
                    
                if len(x_sorted) == 1 :
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split
                
                
                left_indices = data_indices[(x_train[data_indices] < best_split)]
                node_indices[2 * split_node] =left_indices
                is_terminal[2 * split_node]  = False
                need_split[2 * split_node] = True

              
                right_indices = data_indices[(x_train[data_indices] >= best_split)]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1]  =True
                
    return node_splits,node_means,is_terminal



def prediction(x, node_splits, node_means, is_terminal):
    k=1
    while True:
        if is_terminal[k] == True:
            return node_means[k]
        if x>node_splits[k]:
            k=k*2 + 1 
        else:
            k=k*2 
            
            


# In[3]:


P=30

node_splits,node_means,is_terminal = tree(x_train,y_train,P)

y_train_pred= np.array([prediction(x,node_splits,node_means,is_terminal) for x in x_train])
y_test_pred = np.array([prediction(x,node_splits,node_means,is_terminal) for x in x_test])
data_interval = np.arange(0,2.0,0.001)
rectangles=np.array([prediction(data_interval[i],node_splits,node_means,is_terminal) for i in range(len(data_interval))])


plt.figure(figsize=(15,5))
plt.plot(x_train,y_train,"b.", label="training",markersize=10)
plt.plot(data_interval,rectangles,color="black")
plt.ylim(bottom=-1.0)
plt.ylim(top=2.0)
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(loc="upper right")
plt.show()



plt.figure(figsize=(15,5))
plt.plot(x_test,y_test,"r.", label="test",markersize=10)
plt.plot(data_interval,rectangles,color="black")
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.ylim(bottom=-1.0)
plt.ylim(top=2.0)
plt.legend(loc="upper right")
plt.show()




# In[4]:


def rmse(y_truth,y_pred):
    return np.sqrt(sum((y_truth-y_pred)**2)/y_truth.shape[0])

rmse_train=rmse(y_train,y_train_pred)
rmse_test=rmse(y_test,y_test_pred)

print("RMSE on training set is",rmse(y_train,y_train_pred),"when P is",  P)
print("RMSE on test set is",rmse(y_test,y_test_pred),"when P is",  P)


# In[5]:


rmse_test = []
rmse_train=[]
for P in range(10,55,5):
    node_splits,node_means,is_terminal = tree(x_train,y_train,P)
    y_train_pred=np.array([prediction(x,node_splits,node_means,is_terminal) for x in x_train])
    y_test_pred = np.array([prediction(x,node_splits,node_means,is_terminal) for x in x_test])
    rmse_train.append(rmse(y_train,y_train_pred))
    rmse_test.append(rmse(y_test,y_test_pred))
                      
rmse_train=np.array(rmse_train)                     
rmse_test = np.array(rmse_test)
                      
fig = plt.figure(figsize=(15,5))
plt.plot(range(10,55,5),rmse_train,color= "b",label="training",marker="o")
plt.plot(range(10,55,5),rmse_test,color= "r",label="test",marker="o")


plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.legend(loc="upper left")
plt.show()


# In[ ]:




