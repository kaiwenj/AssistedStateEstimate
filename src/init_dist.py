import numpy as np


def gen_init_dist(labels):
    label_prop=np.bincount(labels,minlength=10)/len(labels)
    init_dist=np.array(label_prop)
    return init_dist

def gen_dyn_dist():
    return np.array([np.identity(10)]*11)



