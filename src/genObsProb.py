import numpy as np

def genObservationProbAll(patterns,dataset,state):
    t_img, t_label = dataset
    index = np.invert(np.isnan(patterns))
    patrn = patterns[index]
    subImg = t_img[np.where(t_label == state)]
    numImg = len(subImg)
    count = sum([1 for j in range(numImg) if np.array_equal(subImg[j,:,:][index], patrn)])
    prob=count/numImg
    return prob


def genObservationProbAll(pattern, dataset, state):
    images, labels = dataset
    imagesFromState = images[np.where(labels == state)]
    count = np.mean(np.all(np.all((pattern == imagesFromState) | (np.isnan(pattern)), axis=2), axis = 1)))
    return prob


