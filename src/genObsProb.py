def genObservationProbAll(patterns,rowIndex,dataset):
    t_img, states = dataset
    len_state=len(states)
    prob=np.empty(len_state)
    for i in range(len_state):
        subImg = t_img[np.where(states == states[i])]
        numImg = len(subImg)
        count = sum([1 for j in range(numImg) if np.array_equal(subImg[j, rowIndex, :], patterns)])
        prob[i]=count/numImg
    return prob


