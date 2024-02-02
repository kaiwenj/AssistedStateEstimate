# Sample from belief

def sample_initial(init_dist):
    # Input: 1-D array that represents probability distribution
    # Output: Random index sampled using the probability distribution
    return np.random.choice(np.arange(0, init_dist.size), p=init_dist)

# Sample from observation


# ...
