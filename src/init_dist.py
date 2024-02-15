import numpy as np


def generateInitialDistribution(labels,minlength=10):
    label_proportion_img=np.bincount(labels,minlength=minlength)/len(labels)
    init_dist=np.array(label_proportion_img)
    return init_dist

def generateDynamicDistribution(labels=10,actions=11):
    return np.array([np.identity(labels)]*actions)


def generateObservationProb(current_img,patterns,dataset):
    img,label=current_img
    t_img,t_label=dataset
    subImg=t_img[np.where(t_label == label)]
    numImg=len(subImg)
    patrn=img[patterns,:]
    count = sum([1 for i in range(numImg) if np.array_equal(subImg[i, patterns, :], patrn)])
    return count/numImg

def generateInitialDistributionImg(sample_img):
    img_len=len(sample_img)
    img_proportion=np.ones(img_len)/img_len
    return img_proportion

def generateDynamicDistributionImg(current_img,next_img):
    return int((np.array_equal(current_img,next_img)))

def generateObservationProbImg(patterns,current_img):
    img_size=current_img.shape[0]
    return len(patterns)/img_size

