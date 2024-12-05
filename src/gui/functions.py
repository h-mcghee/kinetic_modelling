import numpy as np 

def background_subtract(x,y,lb, ub):
    mask = (x > lb) & (x < ub)
    mean_y = np.mean(y[mask])
    y -= mean_y
    return y