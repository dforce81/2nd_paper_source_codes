# 2nd paper 3. Integer Nelder-Mead Algorithm


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


#Define x(n_pv), y(n_bat) solutions(coordinates) as a vector
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


# Define sign(sgnd)function
def sgnd(k):
    if k > 0 :
        return 1
    if k == 0 :
        return 0
    else :
        return -1
    
# Initialize dictionary to store results
d_loss_mid = {}
d_loss_xr = {}
d_loss_c = {}
d_loss_s = {}

good = {}
best = {}    
worst = {}

best_x = {}
best_y = {}
good_x = {}
good_y = {}
worst_x = {}
worst_y = {}



## Start N-M algorithm
# Generate three inital random solutions (v1, v2, v3) and run simulation model.
v1 = Vector(2, 3)
n_pv = v1.cx()
n_bat = v1.cy()
exec(open("Simulation.py").read(), globals()) # run Simulation.py with the solution v1
d_loss = d_loss_sim
f_v1 = f(v1.c())
d_loss_v1 = d_loss

v2 = Vector(20, 2)
n_pv = v2.cx()
n_bat = v2.cy()
exec(open("Simulation.py").read(), globals()) # run Simulation.py with the solution v2
d_loss = d_loss_sim
f_v2 = f(v2.c())
d_loss_v2 = d_loss

v3 = Vector(60, 3)
n_pv = v3.cx()
n_bat = v3.cy()
exec(open("Simulation.py").read(), globals()) # run Simulation.py with the solution v3
d_loss = d_loss_sim
f_v3 = f(v3.c())
d_loss_v3 = d_loss


# Initial assign points v1, v2, v3 to points Best, Good, worst points
adict_init = {v1 : [f_v1, d_loss_v1], v2 : [f_v2, d_loss_v2], v3: [f_v3, d_loss_v3]}    
points_init = sorted(adict_init.items(), key=lambda x: x[1])

b = points_init[0][0]
f_b = points_init[0][1][0]
d_loss_b = points_init[0][1][1]
print(" ")
print("Result of the initial best solution: ", b)
print("Result of the initial minimum obj.value at best solution: ", f_b)

g = points_init[1][0]
f_g = points_init[1][1][0]   
d_loss_g = points_init[1][1][1]

wo = points_init[2][0]
f_wo = points_init[2][1][0]
d_loss_wo = points_init[2][1][1]

best_x[0] = b.cx() 
best_y[0] = b.cy() 
good_x[0] = g.cx() 
good_y[0] = g.cy() 
worst_x[0] = wo.cx() 
worst_y[0] = wo.cy() 
   

#Define parameters for the integer N-M algorithm
alpha=2
beta=2
gamma=1
delta=0.4
maxiter = np.arange(15000)+1

mid_dict={}
xr_dict={}
xe_dict={}
cont_dict={}
shrink_dict={}


for i in maxiter:

    mid_x = (g.cx()+b.cx())/2 
    mid_y = (g.cy()+b.cy())/2    
    mid = Vector(mid_x, mid_y)
    mid_dict[i] = mid
            
    n_pv = mid.cx()
    n_bat = mid.cy()

    exec(open("Simulation.py").read(), globals())
    d_loss = d_loss_sim
    f_mid = f(mid.c())
    d_loss_mid = d_loss

    # Define mu = ceil(abs(f_midpoint-f_worst)) and variable k for sgnd(k) function
    mu_x = math.ceil(math.sqrt((mid.cx()-wo.cx())**2)) 
    mu_y = math.ceil(math.sqrt((mid.cy()-wo.cy())**2)) 
    
    k_x = mid.cx()- wo.cx()
    k_y = mid.cy()- wo.cy()

    # Reflection
    xr_x = wo.cx() + alpha*mu_x*sgnd(k_x)
    xr_y = wo.cy() + alpha*mu_y*sgnd(k_y)
    xr = Vector(xr_x, xr_y)
    xr_dict [i] = xr
    
    n_pv = xr.cx()
    n_bat = xr.cy()
    
    exec(open("Simulation.py").read(), globals())
    d_loss = d_loss_sim
    f_xr = f(xr.c())
    d_loss_xr = d_loss


    if f_xr < f_g: # case (i) {either reflect or extend}
        
        if f_b < f_xr : # f_b < f_xr < f_g
            
            wo = xr # replace w with r -> BGR
            f_wo = f_xr
            d_loss_wo = d_loss_xr
            
        else : # f_xr < f_b < f_g
            # Expansion
            xe_x = xr.cx() + beta*mu_x*sgnd(k_x)
            xe_y = xr.cy() + beta*mu_y*sgnd(k_y)
            xe = Vector(xe_x, xe_y)
            xe_dict[i] = xe
            
            n_pv = xe.cx()
            n_bat = xe.cy()

            exec(open("Simulation.py").read(), globals())
            d_loss = d_loss_sim
            f_xe = f(xe.c())        
            d_loss_xe = d_loss

            if f_xe < f_xr : # f_xe < f_xr < f_b
                wo = xe # replace w with e -> BGE
                f_wo = f_xe
                d_loss_wo = d_loss_xe

            else : # f_b < f_xe or f_e < f_b
                wo = xr # replace w with r -> BGR
                f_wo = f_xr
                d_loss_wo = d_loss_xr
 
    else:  # f(g.c()) < f(xr.c()): # case (ii) {either contract or shrink}
        if f_xr < f_wo :
            
            wo = xr
            f_wo = f_xr
            d_loss_wo = d_loss_xr
       
        # Contraction
        c_x = xr.cx() - gamma*mu_x*sgnd(k_x)
        c_y = xr.cy() - gamma*mu_y*sgnd(k_y)
        c = Vector(c_x, c_y)
        cont_dict[i] = c    
        n_pv = c.cx()
        n_bat = c.cy()

        exec(open("Simulation.py").read(), globals())
        d_loss = d_loss_sim
        f_c = f(c.c()) 
        d_loss_c = d_loss
  

        if f_c < f_wo:
            wo = c # replace w with c -> BGC
            f_wo = f_c
            d_loss_wo = d_loss_c

        else :
            # Shrink toward best solution     
            s_x = math.ceil((wo.cx()+b.cx())*delta) # compute s and f(s)
            s_y = math.ceil((wo.cy()+b.cy())*delta)
            s = Vector(s_x, s_y)
            shrink_dict[i] = s
            n_pv = s.cx()
            n_bat = s.cy()
            
            exec(open("Simulation.py").read(), globals())
            d_loss = d_loss_sim
            f_s = f(s.c())         
            d_loss_s = d_loss

            wo = s # replace w with s 
            f_wo = f_s
            d_loss_wo = d_loss_s

            mid_x = math.ceil((g.cx()+b.cx())*delta) 
            mid_y = math.ceil((g.cx()+b.cx())*delta) 
            mid = Vector(mid_x, mid_y)
            mid_dict[i] = mid
                    
            n_pv = mid.cx()
            n_bat = mid.cy()

            exec(open("Simulation.py").read(), globals())        
            d_loss = d_loss_sim
            f_mid = f(mid.c())
            d_loss_mid = d_loss
         
            g = mid # replace g with mid -> BMS
            f_g = f_mid
            d_loss_g = d_loss_mid

 
    # assign points after ith iteration   
    v1 = b
    f_v1 = f_b
    d_loss_v1 = d_loss_b
       
    v2 = g
    f_v2 = f_g
    d_loss_v2 = d_loss_g

    v3 = wo
    f_v3 = f_wo
    d_loss_v3 = d_loss_wo

    
    # update points( b: Best, g: Good, wo : Worst) after ith iteration     
    adict_after = {v1 : [f_v1, d_loss_v1], v2 : [f_v2, d_loss_v2], v3 : [f_v3, d_loss_v3]}      
    points_after = sorted(adict_after.items(), key=lambda x: x[1])
    
    b = points_after[0][0]
    f_b = points_after[0][1][0]
    d_loss_b = points_after[0][1][1]

    g = points_after[1][0]
    f_g = points_after[1][1][0]
    d_loss_g = points_after[1][1][1]
    
    wo = points_after[2][0]
    f_wo = points_after[2][1][0]
    d_loss_wo = points_after[2][1][1]
    
    best[i] = (b, f_b, d_loss_b)
    good[i] = (g, f_g, d_loss_g)
    worst[i] = (wo, f_wo, d_loss_wo)
           
    best_x[i] = b.cx() 
    best_y[i] = b.cy() 
    good_x[i] = g.cx() 
    good_y[i] = g.cy() 
    worst_x[i] = wo.cx() 
    worst_y[i] = wo.cy() 

    print(" ")
    print("Result of best solution of at {i}th iteration: ".format(i=i), best[i][0])
    print("Result of minimum obj.value at best solution at {i}th iteration: ".format(i=i), best[i][1])
    
    
    # Algorithm terminate condition = less than 0.01%
    if abs(1 - (best[i][1] / worst[i][1])) < 0.0001:
        
        break
  

# Display results
print(" ")
print("Result of Nelder-Mead algorithm (aftet {i} th iteration) : ".format(i=i))
print(" ")
print("Best soluito in number of PV array and battery module (make the obj. value minimize) is: ", b)
print(" ")
print("Best soluito in capacities of PV array and battery (make the obj. value minimize) is: ", "(", b.cx()*0.315, ",",  b.cy()*3.3, ")")
print(" ")
print("Unmet demand for the Best soluiton (make the obj. value minimize) is: ", d_loss_b)
print(" ")
print("Minimum obj. value for the best solution is: ", f_b)
print(" ")
print("Total computational time: ", time.perf_counter() - start)  
os.system("taskkill /f /im  ampl.exe") 


## Draw plot for solution movement
iter_figure = np.arange(i+1)
f, ax = plt.subplots(1,1,figsize=(10,10))

for k in iter_figure:

    ax.plot([best_x[k],good_x[k],worst_x[k], best_x[k]],[best_y[k], good_y[k], worst_y[k], best_y[k]],color = 'black', marker='o')
    plt.xlabel('Number of PV Panels', fontsize=20)
    plt.ylabel('Number of Battery Modules', fontsize=20)
#    plt.legend(loc='upper right', fontsize=14)
    plt.xticks(np.arange(0, 66, step=5), fontsize=20)
    plt.yticks(np.arange(0, 6, step=1), fontsize=20)
    plt.savefig('Figure_neld_AU_1.png', format='png', dpi=300 ,bbox_inches='tight')

plt.show()  