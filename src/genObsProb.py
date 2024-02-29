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
    numberOfImagesFromState = len(imagesFromState)
    count = sum(np.all(np.all((pattern == images) | (np.isnan(pattern)), axis=2), axis = 1)))
    prob=count/numberOfImagesFromState
    return prob


