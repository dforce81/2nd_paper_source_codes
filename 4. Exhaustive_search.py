# 2nd paper 4. Exhaustive Search


# Call the AMPL-Phthon API to Python    
from amplpy import AMPL, DataFrame, Environment

# Call some packages  
import numpy as np
import math

from random import *
import os
import time
start  = time.perf_counter()

import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick
import matplotlib.patches as patches

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FuncFormatter
import numpy as np
%matplotlib inline
plt.rcParams["font.family"] = "serif"



#Define x(x_pv), y(x_bat) coordinates as a vector
class Vector(object):
    def __init__(self, x, y):
        """ Create a vector, example: v = Vector(1,2) """
        self.x = x
        self.y = y

    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector(x, y)

    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Vector(x, y)

    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        return Vector(x, y)

    def c(self):
        return (self.x, self.y)
   
    def cx(self):
        return (self.x)

    def cy(self):
        return (self.y)    


# Variable initialization -> it goes to the Simulation.py to get d_loss based on n_pv and n_bat
global n_pv
n_pv = 0
global n_bat
n_bat = 0

global  C_E_Buy
C_E_Buy = 0.13

# Define objective function
def f(point):
    x, y = point
   
    C_PV = 32.38

    C_BAT = 539.11

    return C_PV*n_pv + C_BAT*n_bat + C_E_Buy*d_loss


# Initialize dictionary to store results
result = {}
best= {}
  
# Define parameters for the exhaustive search
PV_basic = 0.315
BAT_basic = 3.3
    
best_v_PV_bat = Vector(0, 0)
best_f_PV_bat = 1000000
best_d_loss_PV_bat = 0

k=1

PV_capacity = np.arange(6.3,31.6,0.315)
BAT_capacity = np.arange(3.3,33.4,3.3)



for i in PV_capacity:
    for j in BAT_capacity:
  
        x_pv = float(i)
        x_bat = float(j)
        v_PV_bat = Vector(x_pv, x_bat)
    
        exec(open("Simulation.py").read(), globals())    
        d_loss_PV_bat = d_loss_sim
        f_PV_bat = f(v_PV_bat.c())
        
        result[k] = (v_PV_bat, f_PV_bat, d_loss_PV_bat)
        k=k+1
        
        if f_PV_bat < best_f_PV_bat:
            best_f_PV_bat = f_PV_bat
            best_v_PV_bat = v_PV_bat
            best_d_loss_PV_bat = d_loss_PV_bat
       
        best[k] = (best_v_PV_bat, best_f_PV_bat, best_d_loss_PV_bat)

        k=k+1

            
# Display results                        
print(" ")
print("Result of exhaustive search for PV_BAT combination: ")
print(" ")
print("     Best PV array and Battery combination: ", best_v_PV_bat)
print(" ")
print("     AEC of Best PV array and Battery combination: ", best_f_PV_bat)
print(" ")
print("     Unmet demand Best PV array and Battery combination: ", best_d_loss_PV_bat)
print(" ")
print("Total computational time: ", time.perf_counter() - start)  

