# Sample from belief

def sampleDistribution(inititialDistribution):
    # Input: 1-D array that represents probability distribution
    # Output: Random index sampled using the probability distribution
    return np.random.choice(np.arange(0, inititialDistribution.size), p=inititialDistribution)


def sampleConditional(distribution, condition):  # (observation, state)
    # Input: 1-D array that represents probability distribution
    # Output: Random index sampled using the probability distribution
    return sampleDistribution(distribution[condition])

# Sample from observation


# ...
