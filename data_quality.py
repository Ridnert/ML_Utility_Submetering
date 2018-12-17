import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def count_nans(input_time_series):
    countnan = 0
    for k in range(len(input_time_series)):
        if input_time_series[k] < 0:
            countnan = countnan + 1
    return countnan/len(input_time_series)


def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])

### LOADING DATA


# This script will take a path as input and concatenate the CSV-files in the folder 


# For loop different classes
# Path vector [cooling electricity heat hotwater water]
path = []
path_test = []
path.append("D:\Master_Thesis_Data\Data_Big\cooling")
path.append("D:\Master_Thesis_Data\Data_Big\electricity\energy")
path.append("D:\Master_Thesis_Data\Data_Big\heating\energy")
path.append("D:\Master_Thesis_Data\Data_Big\hot_water")
path.append("D:\Master_Thesis_Data\Data_Big\water")



# Define file identifier and initiate some variables
file_identifier = "*.csv"
all_data = pd.DataFrame()
labels   = []
datalist = []
counter = 0
num_files_total= [] 
for i in range(len(path)):
    num_files_total.append(count_files(path[i]))
    print("There are " + str(num_files_total[i]) + " files for class " + str(i))
print("Reading & merging files")


Number_Of_Files_To_Use = min(num_files_total) # number of time series to extract from each class
print(Number_Of_Files_To_Use)

nan_percentage = []

binwidth = 2000
colors = ["b","g","r","c","y"]
# Loop over the classes
plt.figure()
for i in range(len(path)):    
    # Defines label
    label_value = i+1
    counter2 = 0
    lengths = []
    for f in glob.glob(path[i] + "/*" + file_identifier):
        # This loops over all files in the path associated with each class.
        counter=counter+1
        counter2 = counter2+1
        if counter%10 == 0:
            
            print(str(counter) + " out of " + str(Number_Of_Files_To_Use*len(path)) + " files have been merged.")
            
        df = pd.read_csv(f,sep=';',na_values=-1) # Reads the file and stores it as a pandas dataframe.
        df.fillna(value=-1,inplace=True)
        data = df.values
        data = data[:,1]
        lengths.append(len(data))
        nan_percentage.append(count_nans(data))
        labels.append(label_value) # Appends label to a list.
        if counter2 > Number_Of_Files_To_Use:
            break

    y,binEdges=np.histogram(lengths,bins=100)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,colors[i])

    
    #plt.hist(lengths,bins=range(min(lengths), max(lengths) + binwidth, binwidth),color = colors[i],alpha = 0.5)
    plt.pause(0.05)
plt.xlabel("Lengths")
plt.ylabel("Number Of Meters")
blue_patch = mpatches.Patch(color='red', label='Cooling')
green_patch = mpatches.Patch(color='blue', label='Electricity')
red_patch = mpatches.Patch(color='green', label='Heating')
cyan_patch = mpatches.Patch(color='cyan', label='Hot Water')
yellow_patch = mpatches.Patch(color='yellow', label='Cold Water')
plt.legend(handles=[blue_patch,green_patch,red_patch,cyan_patch,yellow_patch])
plt.show()


plt.hist(nan_percentage)
plt.xlabel(" Percentage ")
plt.ylabel(" Number Of Meters ")
plt.show()



