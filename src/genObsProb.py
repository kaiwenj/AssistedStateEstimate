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


np.where(np.array([1,2,3])==1)
np.array([[[1,2,1],
             [2,2,2],
             [7,9,3]],
            [[1,2,1],
            [4,4,4],
            [7,9,3]],
            [[7,7,7],
             [2,2,2],
             [7,9,3]]])[np.where(np.array([1,2,3])==1)]