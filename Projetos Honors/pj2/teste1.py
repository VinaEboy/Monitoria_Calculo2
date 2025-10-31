import projeto2 as p2 
import numpy as np 
import math as m
import matplotlib.pyplot as plt

p2.fd_error(lambda x: m.sin(x), lambda x: m.cos(x), 1, 1e-15, 1e-3, 100)

p2.fd_error(lambda x: np.atan(x),lambda x:1/(1+x**2),1,1e-15,1e-1,100)


