import numpy as np


def genInitDist(labels,minlength=10):
    label_prop=np.bincount(labels,minlength=minlength)/len(labels)
    init_dist=np.array(label_prop)
    return init_dist

def genDyndDist(labels=10,actions=11):
    return np.array([np.identity(labels)]*actions)



