#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random

#solution provided for glove_sgd as glove_solution.
#uses log(n_dn) as in course




def glove_SGD():

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
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save('data/embeddings', xs)





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