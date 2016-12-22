#!/usr/bin/env python3
from scipy.sparse import *
import csv
import numpy as np
import pandas as pd
import sklearn.linear_model as sk
import pickle
import random
import re

def create_results():
    embedding_dim = [5, 30, 50, 100, 200]
    eta = [0.0001, 0.001, 0.01]
    alpha = [0.1, 0.25, 0.5, 0.75, 0.9]
    nmax = [50, 75, 100, 150, 200]
    epochs = [1, 4, 6]
    costs = []
    for emb in embedding_dim:
        for e in eta:
            for a in alpha:
                for nm in nmax:
                    for ep in epochs:
                        flag = str("_emb"+str(emb)+"_eta"+str(e)+"_alpha"+str(a)+"_nmax"+str(nm)+"_epochs"+str(ep))
                        print("doing :",flag)
                        cost = glove_SGD(embedding_dim=emb, eta=e, alpha=a, nmax=nm, epochs=ep,flags=flag, track_losses=0)
                        print("DONE :",flag)
                        costs.append(cost)