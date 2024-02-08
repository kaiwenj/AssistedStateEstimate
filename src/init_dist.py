import numpy as np


def genInitDist(labels,minlength=10):
    label_prop=np.bincount(labels,minlength=minlength)/len(labels)
    init_dist=np.array(label_prop)
    return init_dist

def genDyndDist(labels=10,actions=11):
    return np.array([np.identity(labels)]*actions)

def genObsProb(dat,patterns,dataset):
    img,label=dat
    t_img,t_label=dataset
    subImg=t_img[np.where(t_label == label)]
    numImg=len(subImg)
    patrn=img[patterns,:]
    count=0
    for i in range(numImg):
        if np.array_equal(subImg[i,patterns,:],patrn):
            count=count+1
    return count/numImg



