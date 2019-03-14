import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from cycler import cycler
NUM_COLORS = 20

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step



for M in my_range(5,200,10):
    
    p_c = []
    mu = []
    sigma =[]
    prob=[]
    count=0
    for i in my_range(1,30,0.1):
        
        p_c.append((i)/30) 
      
        prob.append(norm.cdf(np.sqrt(M)*(p_c[count]-0.5)/(p_c[count]*(1-p_c[count]))))
        count += 1; 
        

    plt.plot(p_c,prob, 'r-',alpha = M/200)
#plt.show()



K = 10
p = 0.7
sumres=0
for n1 in range(K+1):
    for n2 in range(K-n1+1):
        sumres += math.pow(p,n1)*math.pow((1-p),n2)*math.factorial(int(K))  /  (math.factorial(int(n1))*math.factorial(int(n2)))

print(sumres)