#!/usr/bin/env python3
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import pickle
import random
import re
from glove_routines import *

import os
import re

def extrac_param(name):
    params = []
    splited = str(name).split("_")
    for i in range(2,np.shape(splited)[0]):
        p = splited[i]
        value = re.findall(r"[+-]?\d+(?:\.\d+)?",p)
        params.append(float(value[0]))
    return params

def choose_parameters(threshold):
    '''
    Test different parameters to see which one gives the better results.
    '''
    embedding_dim = [5, 30, 50, 100, 200]
    for subdir, dirs, files in os.walk('metadata'):
        size = np.shape(files)[0]
        losses = np.zeros(size)
        names = []
        param = []
        i=0
        for file in files:
            f = np.load('metadata/'+str(file))
            losses[i] = f
            names.append(file)
            param.append(extrac_param(file))
            i+=1
    print(np.shape(names)[0], "results found.")
    smallest_loss=np.ones(201)*pow(10,20)
    smallest_idx = np.zeros(201)
    small_lost = []
    small_idx = []
    small_lost_param = []
    for i in range(0,np.shape(losses)[0]):
        p = param[i]
        emb = p[0]
        loss = losses[i]
        if(loss<smallest_loss[emb]):
            smallest_loss[emb]=loss
            smallest_idx[emb] = int(i)
            
    for i in range(0,np.shape(losses)[0]):
        p = param[i]
        emb = p[0]
        loss = losses[i]
        if(loss<threshold*smallest_loss[emb]):
            small_lost.append(loss)
            small_lost_param.append(p)
            small_idx.append(i)

    print(np.shape(small_lost)[0], "small results found (",threshold,"threshold )")

    for i in embedding_dim:
        cu_params = param[int(smallest_idx[i])]
        print("Smallest lost for embeddings",cu_params[0],"is :",smallest_loss[i],"with params:",cu_params)
        print("   and name:",names[int(smallest_idx[i])])
    i=0
    for a in small_idx:
        p = param[a]
        print("   ",param[a], " loss is ", small_lost[i]/smallest_loss[p[0]])
        i+=1

def fix_param(data,idx,value):
    assert ((data[:,idx]==value).any())
    newdata = []
    for i in range(0,np.shape(data)[0]):
        if(data[i,idx]==value):

            newdata.append(data[i])
    #print(newdata)
    return recreate_param(newdata)

def recreate_param(data):
    x = np.shape(data)[0]
    y = np.shape(data)[1]
    param = np.zeros((x,y))
    for i in range(0,x):
        for j in range(0,y):
            param[i,j] = data[i][j]
    return param
from mpl_toolkits.mplot3d import Axes3D
def show_graph():
    embedding_dim = [5, 30, 50, 100, 200]
    for subdir, dirs, files in os.walk('metadata'):
        size = np.shape(files)[0]
        losses = np.zeros(size)
        names = []
        param = []
        i=0
        for file in files:
            f = np.load('metadata/'+str(file))
            losses[i] = f
            names.append(file)
            p = extrac_param(file)
            p.append(f)
            param.append(p)
            i+=1
    print(np.shape(names)[0], "results found.")
    param = recreate_param(param)
    fixed_emb = fix_param(param,0,5)
    fixed_eta = fix_param(fixed_emb,1,0.01)
    xs = fixed_eta[:,2]
    ys = fixed_eta[:,3]
    zs = fixed_eta[:,5]
    plt.figure()
    plt.subplot(121)
    plt.axis([min(xs)*0.9,max(xs)*1.1,min(zs)*0.9,max(zs)*1.1])
    plt.plot(xs,zs,'ro')
    plt.subplot(122)
    plt.xscale('log')
    plt.axis([min(ys)*0.9,max(ys)*1.1,min(zs)*0.9,max(zs)*1.1])

    plt.plot(ys,zs,'yo')