# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:47:10 2018

@author: Debasish Jyotishi
"""

import numpy as np
import random

def kmeans(x,k,typ,n_init=10,max_iter=500,tol=0.0001):
    
    # using uniform initialisation method
    if typ=='uniform':
        
        flabel=np.zeros((n_init,np.size(x,1)))
        error=np.zeros(n_init)
        fmeu=np.zeros((x.shape[0],k,n_init))
            
        for j in range(n_init):
            
            #initialisation
            z=int(np.floor(np.size(x,1)/k))
            label=np.zeros(np.size(x,1),int)
            for i in range(0,k-1):
                label[i*z:(i+1)*z]=i*np.ones(z,int)
                
            label[(k-1)*z:]=(k-1)*np.ones(np.size(x,1)-(k-1)*z,int)
            meu=np.zeros([np.size(x,0),k])
            D=4
            iteration=0
            
            #Iteration
            while (D>tol)&(iteration<max_iter):
                
                meu1=np.copy(meu)
                #centroid calculation
                for i in range(0,k):
                    index=[ind for ind,wh in enumerate(label) if wh==i]
                    if index:
                        meu[:,i]=np.sum(x[:,index],1)/np.size(index)
                        
                #Assignment of new label
                label=[np.argmin([np.inner(xx-mu,xx-mu) for mu in meu.T]) for xx in x.T]
                
                #Termination Condition
                D=np.sum(np.sum(np.power((meu-meu1),2),0))
                iteration+=1
                
            flabel[j,:]=label
            error[j]=D
            fmeu[:,:,j]=meu
        
        label=flabel[np.argmin(error),:].astype(int)
        meu=fmeu[:,:,np.argmin(error)]
        return((label,meu))
    
    # using random initialisation method
    elif typ=='random':
        
        flabel=np.zeros((n_init,np.size(x,1)))
        error=np.zeros(n_init)
        fmeu=np.zeros((x.shape[0],k,n_init))
            
        for j in range(n_init):
        
            #initialisation
            val=random.sample(range(np.size(x,1)),k)
            meu=x[:,val]
            label=[np.argmin([np.inner(xx-mu,xx-mu) for mu in meu.T]) for xx in x.T]
            D=4
            iteration=0
            
            #Iteration
            while (D>tol)&(iteration<max_iter):
                
                meu1=np.copy(meu)
                #centroid calculation
                for i in range(0,k):
                    index=[ind for ind,wh in enumerate(label) if wh==i]
                    if index:
                        meu[:,i]=np.sum(x[:,index],1)/np.size(index)
                        
                #Assignment of new label
                label=[np.argmin([np.inner(xx-mu,xx-mu) for mu in meu.T]) for xx in x.T]
                
                #Termination Condition
                D=np.sum(np.sum(np.power((meu-meu1),2),0))
                iteration+=1
            
            flabel[j,:]=label
            error[j]=D
            fmeu[:,:,j]=meu
        
        label=flabel[np.argmin(error),:].astype(int)
        meu=fmeu[:,:,np.argmin(error)]
        return((label,meu))
    
    # using kmeans++ initialisation method
    elif typ=='kmeans++':
        
        flabel=np.zeros((n_init,np.size(x,1)))
        error=np.zeros(n_init)
        fmeu=np.zeros((x.shape[0],k,n_init))
            
        for j in range(n_init):
            
            #initialisation
            val=random.sample(range(np.size(x,1)),1)
            meu=x[:,val]
            for i in range(k-1):
                pr=np.array([np.min([np.inner(xx-mu,xx-mu) for mu in meu.T]) for xx in x.T])
                prob=pr/np.sum(pr)
                innd=np.where(np.cumsum(prob)>np.random.rand(1))
                inter=[[x[0,innd[0][0]]],[x[1,innd[0][0]]]]
                meu=np.concatenate((meu,inter),1)
                
            label=[np.argmin([np.inner(xx-mu,xx-mu) for mu in meu.T]) for xx in x.T]
            D=4
            iteration=0
            
            #Iteration
            while (D>tol)&(iteration<max_iter):
                
                meu1=np.copy(meu)
                #centroid calculation
                for i in range(0,k):
                    index=[ind for ind,wh in enumerate(label) if wh==i]
                    if index:
                        meu[:,i]=np.sum(x[:,index],1)/np.size(index)
                        
                #Assignment of new label
                label=[np.argmin([np.inner(xx-mu,xx-mu) for mu in meu.T]) for xx in x.T]
                
                #Termination Condition
                D=np.sum(np.sum(np.power((meu-meu1),2),0))
                iteration+=1
        
            flabel[j,:]=label
            error[j]=D
            fmeu[:,:,j]=meu
        
        label=flabel[np.argmin(error),:].astype(int)
        meu=fmeu[:,:,np.argmin(error)]
        return((label,meu))
