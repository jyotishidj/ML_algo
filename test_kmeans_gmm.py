# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:54:19 2018

@author: Debasish Jyotishi
"""

import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans
from sklearn.cluster import KMeans
from gmm import gmm
N=100
r=100

#generate data inside the circle 

x=np.zeros((2,N,5))
for i in range(5):
    x[0,:,i]=np.multiply(r*np.random.rand(N),np.cos(2*np.pi*(np.random.rand(N))))
    x[1,:,i]=np.multiply(r*np.random.rand(N),np.cos(2*np.pi*(np.random.rand(N))))

c1=np.zeros((2,N))
c2=np.zeros((2,N))
c3=np.zeros((2,N))
c4=np.zeros((2,N))
c5=np.zeros((2,N))
    
c1[0,:]=x[0,:,0]+100
c1[1,:]=x[1,:,0]+100
c2[0,:]=x[0,:,1]+100
c2[1,:]=x[1,:,1]-100
c3[0,:]=x[0,:,2]-100
c3[1,:]=x[1,:,2]+100
c4[0,:]=x[0,:,3]-100
c4[1,:]=x[1,:,3]-100
c5=x[:,:,4]

plt.figure(0)
plt.scatter(c1[0,:],c1[1,:],10)
plt.hold(True)
plt.scatter(c2[0,:],c2[1,:],20)
plt.scatter(c3[0,:],c3[1,:],30)
plt.scatter(c4[0,:],c4[1,:],40)
plt.scatter(c5[0,:],c5[1,:],50)

y=np.concatenate((c1,c2,c3,c4,c5),1)
np.random.shuffle(y.T)

(cluster1,mean1)=kmeans(y,5,'kmeans++')
label=np.array(cluster1)

cluster2=KMeans(n_clusters=5).fit(y.T)
label1=cluster2.labels_

(cluster2,mean1)=gmm(y,5)
label2=np.array(cluster2)


plt.figure(1)
for i in range(5):
    plt.scatter(y[0,list(np.where(label==i)[0])],y[1,list(np.where(label==i)[0])],(i+1)*10)
    plt.hold(True)
  
plt.figure(2)
for i in range(5):
    plt.scatter(y[0,list(np.where(label1==i)[0])],y[1,list(np.where(label1==i)[0])],(i+1)*10)
    plt.hold(True)
    
plt.figure(3)
for i in range(5):
    plt.scatter(y[0,list(np.where(label2==i)[0])],y[1,list(np.where(label2==i)[0])],(i+1)*10)
    plt.hold(True)