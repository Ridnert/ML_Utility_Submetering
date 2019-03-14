import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
contents = np.loadtxt("probabilities.txt")
contents3class = np.loadtxt('probabilities3.txt')
contents4class = np.loadtxt("probabilities4class.txt")
print(np.shape(contents))
print(np.shape(contents3class))
print(np.shape(contents4class))
maxK = 50
number_of_swipes = 5
gurk = np.arange(1,51)
count = 0
x3= []
for k in range(len(gurk)):
        if gurk[k]%3 !=0:
                x3.append(k+1)
                count = count+1

print(x3)
number_of_eps1_swipes = 10

x = np.arange(1,2*maxK + 1,2)
y_WC = contents


y3class = []
y4class = []
for i in range(number_of_swipes):
    y3class.append(contents3class[(i)*maxK:(i+1)*maxK])
for k in range(number_of_eps1_swipes):
    for i in range(number_of_swipes):
        y4class.append(contents4class[k*number_of_swipes+i*maxK:k*number_of_swipes+(i+1)*maxK])

#y0   =  contents[maxK:2*maxK]
#y1   = contents[2*maxK:3*maxK]
#y2   = contents[3*maxK:4*maxK]

#diff = y[0]-y_WC



## PLOTS

red_line = mlines.Line2D([], [], color='red', label='Lower Bound')
blue_liney0 = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Three Classes, less spread',alpha = 0.3)


blue_liney2 = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Three Classes, more spread',alpha = 1) 

yellow_line1 = mlines.Line2D([], [], color='yellow', label='Four Classes, less spread',alpha=0.3)
yellow_line2 = mlines.Line2D([], [], color='yellow', label='Four Classes, more spread',alpha=1)                       

plt.figure(1)
plt.plot(gurk,y_WC,'r')
for k in range(number_of_swipes):
    plt.plot(gurk,y3class[k],'b',alpha = k/number_of_swipes)

    plt.plot(gurk,y4class[k],'y',alpha = k/number_of_swipes)
plt.legend(handles=[red_line,blue_liney0,blue_liney2,yellow_line1,yellow_line2])
plt.xlabel("K")
plt.ylabel("P[N1 > max(N_i)]")
plt.show()