# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:18:45 2018

@author: Debasish Jyotishi
"""

import numpy as np
from kmeans import kmeans
from scipy.stats import multivariate_normal

def gmm(x,ncluster,thresh=0.0001,max_iter=200):
    
    (label,mean)=kmeans(x,ncluster,'kmeans++',n_init=3,max_iter=500,tol=0.0001)
    var=np.zeros((np.size(x,0),np.size(x,0),ncluster))
    weight=np.zeros((ncluster,1))
    
    for i in range(ncluster):
        index=[ind for ind,wh in enumerate(label) if wh==i]
        if index:
            var[:,:,i]=np.matmul((x[:,index]-mean[:,i].reshape((np.size(x,0),1))),(x[:,index]-mean[:,i].reshape((np.size(x,0),1))).T)/np.size(index)
            weight[i]=np.size(index)/np.size(x,1)
        else:
            var[:,:,i]=np.identity(np.size(x,0))
            weight[i]=np.size(index)/np.size(x,1)
            
    error=4
    iteration=0
    prob=np.zeros((ncluster,np.size(x,1)))
    approb=np.zeros((ncluster,np.size(x,1)))
    
    for i in range(ncluster):
        prob[i,:]= multivariate_normal.pdf(x.T,mean[:,i],var[:,:,i])
    prior1=np.max(prob,0)
    
    while (error>thresh)&(iteration<max_iter):
        
        const=np.sum(np.multiply(np.matlib.repmat(weight,1,np.size(x,1)),prob),0)
        for i in range(ncluster):
            approb[i,:]=np.divide(prob[i,:]*weight[i],const)
           
        #reestimation
        weight=sum(approb,1)/np.size(x,1)
        for i in range(ncluster):
            mean[:,i]=np.sum(np.multiply(np.matlib.repmat(approb[i,:],np.size(x,0),1),x),1)/np.sum(approb[i,:])
            var[:,:,i]=np.matmul(np.matmul((x-mean[:,i].reshape((np.size(x,0),1))),np.diag(approb[i,:])),(x-mean[:,i].reshape((np.size(x,0),1))).T)/np.sum(approb[i,:])  
            
        for i in range(ncluster):
            prob[i,:]= multivariate_normal.pdf(x.T,mean[:,i],var[:,:,i])
        
        #condition check
        prior=np.max(prob,0)
        error=np.absolute(sum(prior-prior1))/np.size(x,1)
        iteration+=1
        
        prior1=prior
        
    label=np.argmax(prob,0)
    return(label,(weight,mean,var))

    