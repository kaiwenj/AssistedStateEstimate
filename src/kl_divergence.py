import numpy as np

# Calculate KL Divergence
def calculateKLDivergence(p, q):
    qShift = (q + 10**-6)
    qShift = qShift/sum(qShift)
    pShift = (p + 10 ** -6)
    pShift = pShift / sum(qShift)
    return np.sum(pShift * np.log(pShift / qShift))

# Give observation that minimizes KL Divergence
def sythesizeObs(machineBelief, agentBeliefGivenObs, obsSpace):
    # machineBelief: Machine's belief distribution after observing the true state
    # agentBeliefGivenObs: Agent's belief distribution conditioned on machine given observation
    # Return: Machine returns observation that changes agent's belief to most closely match the machine's belief
    klDivGivenObs = np.array([calculateKLDivergence(machineBelief,agentBeliefGivenObs[o]) for o in obsSpace])
    return np.argmin(klDivGivenObs)

