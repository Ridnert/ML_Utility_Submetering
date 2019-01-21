import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from collections import deque
from numpy import linalg as LA
def GetShiftingWindows(thelist, size):
    return [ thelist[x:x+size] for x in range( len(thelist) - size + 1 ) ]

## In this code I will try and implement the data dictionary slice selection algorithm described in paper
# https://www.cs.ucr.edu/~eamonn/SDM_RealisticTSClassifcation_cameraReady.pdf

def shuffle_data(input_data):
   # np.random.seed(seed=2018)
    index  = np.arange(0,np.shape(input_data)[1])
    choice = np.random.choice(index,np.shape(input_data)[1],replace=False)
    output_data = np.zeros(np.shape(input_data))
    output_data   = input_data[:,choice]
    return output_data

def NN_search(q,T):
    ## q:query T a time series
    w = GetShiftingWindows(T,len(q))
    print(w[0])
    dist_vector = np.zeros([1,len(w)])
    for i in range(len(w)):
        dist_vector[0,i] = LA.norm(q-w[i])
    NN_dist = np.min(dist_vector)
    return dist_vector,NN_dist

def classification(D,labels,r,q):
    # D data dictionary
    bsf = 1e9
    class_label = 0
    for i in range(len(D)):
        dist = NN_search(q,D[i])
        if dist < bsf:
            bsf = dist
            class_label = labels[i]
    NN_dist = bsf
    if NN_dist > r:
        return -1
    else
        return class_label

# Importing the concatenated data
path = "D:\Master_Thesis_Data\Concatenated_File_total.csv"
#Hard Threshholding for removing outliers
df = pd.read_csv(path,sep=';',header=None)
data = df.values
shape = np.shape(data)
data = shuffle_data(data)


test = np.asarray([1,2,3,4,5,6,7,8,9,10,11,10])
o = np.asarray([6,7])
out = NN_search(o,test)
print(out)

q = "random selected subsequence from data"
A = np.where("class of training data = class of q")
B = np.where("class of training data = class of q")

dists_A = []
dists_B = []

bsf = 1e9

for i in range(len(A)):
    dist_vector,NN_dist = NN_search(q,A[i])
    if NN_dist < bsf:
        bsf = NN_dist
    dists_A.append(dist_vector)

NN_friends_dist = bsf
bsf = 1e9

for j in range(len(B)):
    dist_vector2,NN_dist2 = NN_search(q,B[j])
    if NN_dist2 < bsf:
        bsf = NN_dist
    dists_B.append(dist_vector2)
NN_enemy_dist = bsf

if NN_friend_dist < NN_enemy_dist:
    likely_true_positives = np.where(dists_A < NN_enemy_dist)
else:
    likely_false_positives = np.where(dists_B < NN_friends_dist)
