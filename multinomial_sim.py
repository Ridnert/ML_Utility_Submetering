
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines



## This code shall draw some values from multinomial distribution
# We vary the probability distribution in order to see the effects
p= 0.7
randomperm = False# Set this to false for noew
p1 = [p,0.3]

p2 = [p,0.25,0.05]
p3 = [p,0.2,0.1]
p4 = [p,0.2,0.05,0.05]
p5 = [p,0.15,0.05,0.05,0.05]

P =[]
P.append(p1)
P.append(p2)
P.append(p3)
P.append(p4)
P.append(p5)


range_K = 50

colors = plt.cm.jet(np.linspace(0,1,6))
num_repeats = 50000
final_accuracy = np.zeros([len(P),range_K])
prob = np.zeros(len(P[0]))
for C in range(len(P)):
   
    count = 0
    for K in range(1,range_K+1):
        if(randomperm == False):
            vals = np.random.multinomial(K,P[C],size = num_repeats)
        preds =[]
        
        for p in range(num_repeats):
            # When highest number is shared among two or more classes
            if(randomperm == True):
                eps = np.random.uniform(0,0.3)
                #random_disturb = [-eps,eps,0,0,0]
                prob[:] = P[C]
                prob[0] = prob[0] - eps
                prob[1] = prob[1] + eps
                
                vals = np.random.multinomial(K,prob,size = 1)
                max_value = np.max(vals)
                c = np.where(vals[0,:] == max_value)
                if len(c[0] > 1):
                    c = np.random.choice(c[0])
                preds.append(c)
            else:
                max_value = np.max(vals[p,:])
                c = np.where(vals[p,:] == max_value)
                if len(c[0] > 1):
                    c = np.random.choice(c[0])
                preds.append(c)

        gurk,ri,rc = np.unique(preds,return_index = True,return_counts = True)
        final_accuracy[C,count]  =(rc[0]/num_repeats)
      
        count=count+1
        

p1_line = mlines.Line2D([], [], color=colors[0], label='P1 = [0.7,0.3]')
p2_line = mlines.Line2D([], [], color=colors[1], label='P2 = [0.7,0.25,0.05]')
p3_line = mlines.Line2D([], [], color=colors[2], label='P3 = [0.7,0.2,0.1]')
p4_line = mlines.Line2D([], [], color=colors[3], label='P4 = [0.7,0.2,0.05,0.05]')
p5_line = mlines.Line2D([], [], color=colors[4], label='P5 = [0.7,0.15,0.05,0.05,0.05]')
x = np.arange(1,range_K+1)
for k in range(len(P)):
    plt.plot(x,final_accuracy[k,:],color=colors[k])

plt.legend(handles=[p1_line,p2_line,p3_line,p4_line,p5_line])
plt.xlabel('K')
plt.ylabel('Probability')


plt.show()
