import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# This Code Plots the performance as slice size increases
# SA: Slice Accuracy, VA: Voting Accuracy
x = [24,32,48,72,100] # Slice Size

VA_BNN = [67.2, 62.7, 75.0, 78.8, 67.3]
SA_BNN = [63.7, 57.2,67.6, 75.8, 61.9]

VA_DNN = [64.2, 65.7, 73.9, 75.6, 65.3]
SA_DNN = [64.0, 56.6, 68.5, 73.1, 59.6]


plt.figure()
red_line_VA = mlines.Line2D([], [], color='red', label='BNN')
red_line_SA = mlines.Line2D([], [], color='red', label='Lower Bound',alpha=0.5)

blue_line_VA = mlines.Line2D([], [], color='blue', label='DNN')
blue_line_SA = mlines.Line2D([], [], color='blue', label='DNN')
plt.plot(x,VA_BNN,'r')
plt.plot(x,SA_BNN,'r',alpha = 0.5)
plt.plot(x,VA_DNN,'b')
plt.plot(x,SA_DNN,'b',alpha=0.5)
plt.legend(handles = [red_line_VA,red_line_SA,blue_line_VA,blue_line_SA])
plt.show()

