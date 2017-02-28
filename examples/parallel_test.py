from joblib import Parallel, delayed
import numpy as np
from math import sqrt
def my_function(num):

    return num * num



array = ( np.arange(1000)+5 ) * 3



print Parallel(n_jobs=2)(delayed(my_function)(i) for i in array)
