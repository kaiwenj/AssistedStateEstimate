import numpy as np

# Calculate KL Divergence
def KLDivergence(p, q):
    qShift = q + 10**-6
    return np.sum(np.where(p != 0, p * np.log(p / qShift), 0))

# Give observation that minimizes KL Divergence
def sythesizeObs(trueDist, condDist, obsSpace): #here the indez of dist. conditions the dist for given 0
    # Before, the dist was conditioned
    klDivGivenObs = np.array([KLDivergence(trueDist,condDist[o]) for o in obsSpace])
    return np.argmin(klDivGivenObs)
