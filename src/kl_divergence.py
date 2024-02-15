import numpy as np

# Calculate KL Divergence
def KLDivergence(p, q):
    qShift = (q + 10**-6)
    qShift = qShift/sum(qShift)
    pShift = (p + 10 ** -6)
    pShift = pShift / sum(qShift)
    return np.sum(pShift * np.log(pShift / qShift))

# Give observation that minimizes KL Divergence
def sythesizeObs(trueDist, condDist, obsSpace): #here the indez of dist. conditions the dist for given 0
    # Before, the dist was conditioned
    klDivGivenObs = np.array([KLDivergence(trueDist,condDist[o]) for o in obsSpace])
    return np.argmin(klDivGivenObs)
