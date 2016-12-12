#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random

def update_cost(cost, fn, x, w, z, newW, newZ):
    cost = cost - 1/2 * fn* pow(x - np.dot(w,z),2)
    cost = cost + 1/2 * fn* pow(x - np.dot(newW, newZ),2)
    return cost

def glove_SGD(embedding_dim = 20, eta = 0.001, alpha = 3 / 4, nmax = 100, epochs = 10, track_losses = False, flags=""):
    '''
    Create the embeddings.npy (and the embeddings_cost.npy) needed to perform logistic regression later.
    Params :
        embedding_dim is the size of the vectors associated to each word
        eta is the learning rate
        alpha is the power used in the weight function f which gives importance to each entry. alpha must be between 0 and 1.
        nmax is the diviser used in the weight function f.
        epochs is the number of time the SGD will be perform.
        track_losses must be >0 if the loss must be track. The function will return an array of the evolution of the loss (computed at each iteration i%track_loss==0) if the track_losses>0.
        flags allows to modify the name of the file created by this function     
    Return : If track_losses = -1, nothing, else if track_losses = 0, only the last loss, and if track_loss>0, the evolution of the loss (array)
    '''
    assert track_losses>=-1
    assert alpha>=0
    assert alpha<=1
    assert embedding_dim>0
    assert epochs>0
    print("loading cooccurrence matrix")
    with open('data/cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))
    print("initializing parameters : nmax =",nmax,",cooc.max() =", cooc.max(),", embedding_dim =",embedding_dim,", eta =",eta,", alpha =",alpha,", epochs =",epochs,".")
    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))
    cost = 0
    if(track_losses>=0):
        print("initializing cost")
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            x = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            w, z = xs[ix, :], ys[jy, :]
            WZ = np.dot(w,z)
            cost = cost + fn * pow(x - WZ,2)
        cost = cost/2
        if(track_losses>0):
            losses = []
            losses.append(cost)
    print("Running now")
    i = 0
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            newX = np.copy(scale * y + xs[ix, :])
            newY = np.copy(scale * x + ys[jy, :])
            if(track_losses>=0):
                cost = update_cost(cost,fn,logn,x,y,newX,newY)
            xs[ix, :] = newX
            ys[jy, :] = newY
            if(track_losses>0):
                if(i%track_losses==0):
                    losses.append(cost)
    np.save(str('data/embeddings'+flags), xs)
    if(track_losses>0):
        np.save(str('metadata/embeddings_cost'+flags),losses)
        return losses
    if(track_losses==0):
        return cost
    
def glove_template():
    print("NOT WORKING : DO NOT USE YET")
    print("loading cooccurrence matrix")
    with open('data/cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            fn = min(1.0, (n / nmax) ** alpha)
            #cost = fn*np.power(np.dot(xs[ix,:],np.transpose(ys[jy,:]) - fn),2)
            #gradientwi = 2*fn*((np.dot(xs[ix,:],np.transpose(ys[jy,:])-fn)*xs[ix,:]))
            #gradientwj = 2*fn*((np.dot(xs[ix,:],np.transpose(ys[jy,:])-fn)*ys[jy,:]))
            #xs[ix,:] -= gradientwi*xs[ix,:]
            #ys[jy,:] -= gradientwj*ys[jy,:]
    np.save('data/embeddings', xs)