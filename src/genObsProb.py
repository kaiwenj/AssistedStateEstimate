import numpy as np

def genObservationProbAll(pattern, dataset, state):
    images, labels = dataset
    imagesFromState = images[np.where(labels == state)]
    prob = np.mean(np.all(np.all((pattern == imagesFromState) | (np.isnan(pattern)), axis=2), axis = 1))
    return prob


