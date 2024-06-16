import numpy as np

def epsilon_greedy(Q, epsilon, seed=None):
    """
    Epsilon-greedy action selection method.
    
    Parameters:
    Q (numpy array): A numpy array representing the estimated action values.
    epsilon (float): The probability of selecting a random action.
    
    Returns:
    int: The selected action.
    """
    if seed is not None:
        np.random.seed(seed)

    rgn = np.random.default_rng(seed)
    if rgn.random() < epsilon:
        return rgn.integers(len(Q))
    else:
        return np.argmax(Q)

def greedy(Q: np.ndarray) -> int:
    """
    Greedy action selection method.
    
    Parameters:
    Q (numpy array): A numpy array representing the estimated action values.
    
    Returns:
    int: The selected action.
    """
    return np.argmax(Q)