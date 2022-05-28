#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse 
import matplotlib.transforms as transforms

dataset=np.genfromtxt("hw08_data_set.csv", delimiter = ",")
centroids=np.genfromtxt("hw08_initial_centroids.csv", delimiter = ",")


# In[2]:


class_means = np.array([[+5, +5],
                        [-5, +5],
                       [-5,-5],
                       [+5,-5],
                       [+5,+0],
                       [+0,+5],
                       [-5,+0],
                       [+0,-5],
                       [+0,+0]])


# covariance parameters
class_covariances = np.array([[[+0.8, -0.6], 
                               [-0.6, +0.8]],
                              
                              [[+0.8, +0.6], 
                               [+0.6, +0.8 ]],
                              
                             [[+0.8, -0.6],
                             [-0.6, +0.8]],
                             
                             [[+0.8, +0.6],
                             [+0.6, +0.8]],
                             
                            [[+0.2,+0.0],
                            [+0.0,+1.2]],
                             
                            [[+1.2,+0.0],
                            [+0.0,+0.2]],
                            
                            [[+0.2,+0.0],
                            [+0.0,+1.2]],
                             
                            [[+1.2,+0.0],
                             [+0.0,+1.2]],
                              
                            [[+1.6,+0.0],
                              [+0.0,+1.6]]])

K=9 

N=dataset.shape[0]

# sample sizes
class_sizes=np.array([100,100,100,100,100,100,100,100,200])


points1=np.array([np.random.multivariate_normal(mean = class_means[0,:],
                                        cov = class_covariances[0,:,:],
                                        size = class_sizes[0])])



points2=np.array([np.random.multivariate_normal(mean = class_means[1,:],
                                        cov = class_covariances[1,:,:],
                                        size = class_sizes[1])])

points3=np.array([np.random.multivariate_normal(mean = class_means[2,:],
                                        cov = class_covariances[2,:,:],
                                        size = class_sizes[2])])

points4=np.array([np.random.multivariate_normal(mean = class_means[3,:],
                                        cov = class_covariances[3,:,:],
                                        size = class_sizes[3])])

points5=np.array([np.random.multivariate_normal(mean = class_means[4,:],
                                        cov = class_covariances[4,:,:],
                                        size = class_sizes[4])])

points6=np.array([np.random.multivariate_normal(mean = class_means[5,:],
                                        cov = class_covariances[5,:,:],
                                        size = class_sizes[5])])

points7=np.array([np.random.multivariate_normal(mean = class_means[6,:],
                                        cov = class_covariances[6,:,:],
                                        size = class_sizes[6])])

points8=np.array([np.random.multivariate_normal(mean = class_means[7,:],
                                        cov = class_covariances[7,:,:],
                                        size = class_sizes[7])])

points9=np.array([np.random.multivariate_normal(mean = class_means[8,:],
                                        cov = class_covariances[8,:,:],
                                        size = class_sizes[8])])



plt.figure(figsize = (8, 8))
plt.plot(dataset[:,0], dataset[:,1],"k.", markersize = 10)

plt.plot()
plt.xlim((-8, +8))
plt.ylim((-8, +8))
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.show()



# In[3]:


memberships = np.argmin(spa.distance_matrix(centroids, dataset),axis=0)
class_priors = [np.mean(memberships == c) for c in range(K)]
class_sizes = [dataset[memberships == c].shape[0] for c in range(K)]

means = np.array([np.mean(dataset[memberships == c], axis=0) for c in range(K)])
covariances = np.array([(np.transpose(dataset[memberships == c] - means[c]) @ 
                        (dataset[memberships == c] - means[c])) / class_sizes[c] for c in range(K)])

print(class_priors)
print(covariances)


# In[4]:


iterations = 100

for i in range(iterations):
    h1 = np.transpose(np.array([class_priors[c] * multivariate_normal.pdf(dataset, means[c], covariances[c])
                                for c in range(K)]))
    h2 = np.transpose(np.array([h1[:, c] / np.sum(h1, axis=1) for c in range(K)]))

    class_priors = [np.sum(h2[:, c]) / N for c in range(K)]
    means = np.array([h2[:, c] @ dataset / np.sum(h2[:, c]) for c in range(K)])
    covariances = np.array([(h2[:, c] * np.transpose(dataset - means[c]) @ 
                             (dataset - means[c])) / np.sum(h2[:, c]) for c in range(K)])
    memberships = np.argmax(h2, axis=1)

print(means)


# In[15]:


def ellipse(x, y, means, ax, edgecolor, linestyle, std):
    cov = np.cov(x, y)


    radius_x = np.sqrt(1 + cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    radius_y = np.sqrt(1 - cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    ellipse = Ellipse((0, 0), width=radius_x * 2, height=radius_y * 2, facecolor="none",
                      edgecolor=edgecolor, linestyle=linestyle)

   
    scale_x = cov[0, 0] * std
    mean_x = means[0]
    scale_y = cov[1, 1] * std
    mean_y = means[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)



cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

stds=np.array([2.5,2.5,2.5,2.5,1.5,2.5,2.5,2.5,2.5])


fig, ax = plt.subplots(1, 1, figsize=(6, 6))

for c in range(K):
    ellipse(dataset[memberships == c][:, 0], dataset[memberships == c][:, 1], means[c], ax, cluster_colors[c], "-",
            stds[c])
    plt.plot(dataset[memberships == c, 0], dataset[memberships == c, 1], "o", markersize=7, color=cluster_colors[c])

means1 = [np.mean(dataset[memberships == 0, 0]), np.mean(dataset[memberships == 0, 1])]
ellipse(dataset[memberships == 0, 0], dataset[memberships == 0, 1], means1, ax, 'k', '--', stds[0])

means2 = [np.mean(dataset[memberships == 1, 0]), np.mean(dataset[memberships == 1, 1])]
ellipse(dataset[memberships == 1, 0], dataset[memberships == 1, 1], means2, ax, 'k', '--', stds[1])

means3 = [np.mean(dataset[memberships == 2, 0]), np.mean(dataset[memberships == 2, 1])]
ellipse(dataset[memberships == 2, 0], dataset[memberships == 2, 1], means3, ax, 'k', '--', stds[2])

means4 = [np.mean(dataset[memberships == 3, 0]), np.mean(dataset[memberships == 3, 1])]
ellipse(dataset[memberships == 3, 0], dataset[memberships == 3, 1], means4, ax, 'k', '--', stds[3])

means5 = [np.mean(dataset[memberships == 4, 0]), np.mean(dataset[memberships == 4, 1])];
ellipse(dataset[memberships == 4, 0], dataset[memberships == 4, 1], means5, ax, 'k', '--',stds[4])

means6 = [np.mean(dataset[memberships == 5, 0]), np.mean(dataset[memberships == 5, 1])];
ellipse(dataset[memberships == 5, 0], dataset[memberships == 5, 1], means6, ax, 'k', '--', stds[5])


means7 = [np.mean(dataset[memberships == 6, 0]), np.mean(dataset[memberships == 6, 1])];
ellipse(dataset[memberships == 6, 0], dataset[memberships == 6, 1], means7, ax, 'k', '--', stds[6])


means8 = [np.mean(dataset[memberships == 7, 0]), np.mean(dataset[memberships == 7, 1])];
ellipse(dataset[memberships == 7, 0], dataset[memberships == 7, 1], means8, ax, 'k', '--',stds[7])


means9 = [np.mean(dataset[memberships == 8, 0]), np.mean(dataset[memberships == 8, 1])];
ellipse(dataset[memberships == 8, 0], dataset[memberships == 8, 1], means9, ax, 'k', '--', stds[8])




plt.xlim(left=-8)
plt.xlim(right=+8)
plt.ylim(bottom=-8)
plt.ylim(top=+8)
plt.xlabel("$x_{1}$")
plt.ylabel("$x_{2}$")
plt.title("Clustering Results")
plt.show()





# In[ ]:




