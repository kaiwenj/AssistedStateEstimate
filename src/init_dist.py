import numpy as np


def generateInitialDistribution(labels,minlength=10):
    label_proportion_img=np.bincount(labels,minlength=minlength)/len(labels)
    init_dist=np.array(label_proportion_img)
    return init_dist

def generateDynamicDistribution(labels=10,actions=11):
    return np.array([np.identity(labels)]*actions)


def generateObservationProb(current_img,pattern,dataset):
    images, labels = dataset
    s_image, s_label = current_img
    imagesFromState = images[np.where(labels == s_label)]
    prob = np.mean(np.all(np.all((pattern == imagesFromState) | (np.isnan(pattern)), axis=2), axis=1))
    return prob


def generateInitialDistributionImg(sample_img):
    img_len=len(sample_img)
    img_proportion=np.ones(img_len)/img_len
    return img_proportion

def generateDynamicDistributionImg(current_img,next_img):
    return np.sum((np.array_equal(current_img,next_img)))

def generateObservationProbImg(patterns,current_img):
    img_size=current_img.shape
    patrn = np.invert(np.isnan(patterns))
    return np.sum(patrn)/(img_size[0]*img_size[1])

