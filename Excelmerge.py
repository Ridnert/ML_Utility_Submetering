import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
#path.append("D:\Master_Thesis_Data\Data_Big\hot_water")
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
#Number_Of_Files_To_Use = 220

# Loop over the classes
for i in range(len(path)):    
    # Defines label
    label_value = i+1
    counter2 = 0
    for f in glob.glob(path[i] + "/*" + file_identifier):
        # This loops over all files in the path associated with each class.
        counter=counter+1
        counter2 = counter2+1
        if counter%10 == 0:
            
            print(str(counter) + " out of " + str(Number_Of_Files_To_Use*len(path)) + " files have been merged.")
            
        df = pd.read_csv(f,sep=';',na_values=-1) # Reads the file and stores it as a pandas dataframe.
        datalist.append(df) # appends the dataframe to a list
        labels.append(label_value) # Appends label to a list.
        if counter2 > Number_Of_Files_To_Use:
            break
all_data = pd.concat(datalist, axis = 1) # Concatenates the datafiles.
del datalist # Freeing up some memory
del df        
print()
print("Done!")
all_data.fillna(value=-1,inplace=True)
all_data = all_data.values # Extracs the values as array.




num_of_samples = np.shape(all_data)[1]/2
df_time_series = np.zeros([1+np.shape(all_data)[0],int(np.shape(all_data)[1]/2)]) # Plus one for allowing to add label vector
print("Starting to build the time series matrix, disregarding the data of date & time.")


# Looping over the number of samples
for k in range(int(num_of_samples)):
    if k%100 == 0 and k!=0:  
        print(k)
    df_time_series[1:int(np.shape(df_time_series)[0]),k] = all_data[:,2*k+1]


df_time_series[0,:] = np.array([labels]) # adding the label vector as the first row in the data-matrix


filename = "D:\Master_Thesis_Data\Concatenated_File_total"

print("Done!, adding label vector and writing file to: " + filename +".csv")  
if os.path.exists(filename + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
    
    os.remove(filename + ".csv")
np.savetxt(filename + ".csv", df_time_series ,delimiter = ';') # Prints data to file