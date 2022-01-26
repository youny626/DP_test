import pandas as pd
import diffprivlib as dp
import numpy as np

def count(array, epsilon):
    real_count = len(array)
    laplace = dp.mechanisms.Laplace(epsilon=epsilon, sensitivity=1)
    return laplace.randomise(real_count)
    # dp_count = dp.tools.count_nonzero(np.ones_like(array), epsilon)
    # return dp_count
