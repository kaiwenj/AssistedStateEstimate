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


def genInitDistImg(s_img):
    img_len=len(s_img)
    img_prop=np.ones(img_len)/img_len
    return img_prop

def genDynDistImg(s_img,actions=11):
    img_len = len(s_img)
    return np.array([np.identity(img_len)]*actions)

def genObsProbImg(patterns,img_size=28):
    return len(patterns)/img_size


