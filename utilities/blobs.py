import numpy as np 
import matplotlib.pyplot as plt 
import time

def fun1(x:int):
    return x**4+x**3+x**2+x+0.08


def YValue(X, fun):
    Y = np.ones([X.shape[0], 1])
    for i in range(X.shape[0]):
        Y[i, 0] = X[i, 1] < fun(X[i,0])
    return Y

def timeit(func):
    """
    Prints execution time of a function.

    Parameters
    ----------
    func : function
        The function to be measured.
    """
    
    def wrapper(*args, **kwargs):
        global time_elapsed 
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        time_elapsed = end_time - start_time
        
        print(
            f"\n-----------Execution time: {time_elapsed:.3f} seconds--------\n")
        return result
    return wrapper

