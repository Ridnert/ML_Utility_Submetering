
import numpy as np
import pandas as pd

## This code shall draw some values from multinomial distribution
# We vary the probability distribution in order to see the effects

p1 = [0.7,0.3,0,0,0]
p2 = [0.7,0.25,0.05,0,0]
p3 = [0.7,0.2,0.1,0,0]
p4 = [0.7,0.2,0.05,0.05,0]
p5 = [0.7,0.1,0.1,0.1,0]
p6=  [0.7,0.05,0.1,0.05,0.05]
P =[]
P.append(p1)
P.append(p2)
P.append(p3)
P.append(p4)
P.append(p5)
P.append(p6)

# parameters
vals= []

M = 10 ## number of simulations
num_repeats = 100
final_accuracy = np.zeros([6,6])
for C in range(6):
   # vals= []
    preds =[]
    count = 0

    for K in [2,5,10,20,50,100]:
        vals = np.random.multinomial(K,P[C],size=num_repeats)
        preds =[]
        
        for p in range(num_repeats):
            preds.append(np.argmax(vals[p,:]))
        gurk,ri,rc = np.unique(preds,return_index = True,return_counts = True)
        final_accuracy[C,count]  =(rc[0]/num_repeats)
        print(rc)
        count=count+1
        
  
       # print(vals[K])
        
print(final_accuracy)