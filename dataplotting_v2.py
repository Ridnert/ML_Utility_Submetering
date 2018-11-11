import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import interpolate
import os
from sklearn.preprocessing import minmax_scale


# PARAMETERS TO CHANGE
## INTERPOLATIONFUNCTION FOR SLICES

def interp(slice):
    
    # Finding positive indices to build function
    ind = np.where(slice >= 0)
    y = slice[ind]

    
    f = interpolate.interp1d(ind[0],y, fill_value= "extrapolate",kind='quadratic')


    xnew = np.arange(0,np.shape(slice)[0] , 1)

    ynew = f(xnew)
    ynew[np.where(ynew < 0)] = 0 # Avoids extrapolating into negative numbers
    return ynew

def shuffle_data(input_data):
    np.random.seed(seed=2018)
    index  = np.arange(0,np.shape(input_data)[1])
    choice = np.random.choice(index,np.shape(input_data)[1],replace=False)
    output_data = np.zeros(np.shape(input_data))
    output_data   = input_data[:,choice]
    return output_data

def main(time_points):
    np.random.seed(seed=2018)
    random.seed(2018)
    print("Number of time_Points is " + str(time_points))
    number_of_training_slices = 200
    number_of_test_slices     = 100
    zeros_slice_percentage = 0.3


    path = "D:\Master_Thesis_Data\Concatenated_File_total.csv"
    #Hard Threshholding for removing outliers
    df = pd.read_csv(path,sep=';',header=None)
    data = df.values
    shape = np.shape(data)
    # Shuffeling the data
    data   = shuffle_data(data)

    print("Number of Time Series is:" + str(np.shape(data)[1]))
    print( np.any(np.isnan(data)))
    max_length = np.shape(data)[0]
    number_of_classes = len(np.unique(data[0,:]))


    training_percentage = 0.9
    threshhold = 5000              # To remove outliers                         
        # 720 is one month 9000 a year
    indices = data[:,:] < threshhold
    data[indices == False] = -1 

    data_test  = data[:,int(training_percentage*shape[1]):shape[1]] # For writing the test-time series to file

    # Training Data
    y = data[0,0:int(training_percentage*shape[1])]
    X  = data[1:shape[0],0:int(training_percentage*shape[1])]

    # Test Data                                                                           # Shall be written directly to file or not, not
    X_test = data[1:max_length,int(training_percentage*shape[1]):shape[1]]
    y_test = data[0,int(training_percentage*shape[1]):shape[1]]


    ##########################################################################################################
    ############# Next we create the chunks of size time_points which will be stored as columns in ###########
    ############# X_resampled with corresponding label in y_resampled FOR TRAINING SET             ###########
    ############################################################################################333
    number_of_chunks_per_customer = []
    X_resampled = [] # Stores the data
    y_resampled = [] # Stores the label
    meter_resampled = [] # Stores the meter value (Ranging from 0-shape[1])
    left_out_slice = []
    booleans    = X[:,:] >=  0
    booleans2   = X[:,:] ==  0
    boolvec = []
    for j in range(np.shape(X)[1]):                                                    # Looping over the samples
        label = y[j]
        count = 0
        for i in range(int(np.shape(X)[0] / time_points)):  # Looping over the number of chunks assuming no overlap
            # In this statement we take the chunks where all values are greater or equal to zero(I.e excluding the NaN) and where there are at most 25% values equal to zero.
            #Then append them in X_resampled
            # Take 10 non Constant Slices from each sample
    
        #  if count < num_of_training_slices: # Change this to change the number of snippets from each time-series
            if  np.all(booleans[i*time_points:i*time_points+time_points,j]) == True and \
            np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points*zeros_slice_percentage and\
            np.var(X[i*time_points:i*time_points+time_points,j]) > 1e-3:

               

                var = np.var(X[i*time_points:i*time_points+time_points,j]) # Adding some augmentation to the data random noise

                X_resampled.append(X[i*time_points:i*time_points+time_points,j]) #+ np.random.normal(0,np.sqrt(var),np.shape(X[i*time_points:i*time_points+time_points,j])))
                y_resampled.append(label)
                meter_resampled.append(j)

                count = count+1

            elif np.sum(booleans[i*time_points:i*time_points+time_points,j]) > 0.8*time_points and \
                np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points*zeros_slice_percentage and \
                np.var(X[i*time_points:i*time_points+time_points,j]) > 1e-3:
                # Filling NaN's with interpolation if the number of NaN's per slice is below some limit
                

                # Take non Constant Slices

                out = interp(X[i*time_points:i*time_points+time_points,j])
               
                X_resampled.append(out)
                y_resampled.append(label)
                meter_resampled.append(j)
                count = count+1


            else: # Monitor the Slices that are left out
                  
                left_out_slice.append(X[i*time_points:i*time_points+time_points,j])
                boolvec.append(time_points-np.sum(booleans[i*time_points:i*time_points+time_points,j]))

        number_of_chunks_per_customer.append(count)


    plt.hist(boolvec)
    plt.title("Histogram of number of NaN's per slice with the maximum nuber of zeros equal to "+str(int(time_points*zeros_slice_percentage)))
    plt.show()
    left_out_slice = np.stack(left_out_slice,axis=-1)
    plt.plot(number_of_chunks_per_customer)
    plt.title("Distribution of Number of Chunks on Customers/Meters")
    plt.show()

                                        
                                        
                                        

    #######################  Creates dataset of slices for test dataset    ###############################
    number_of_chunks_per_customer_test = []
    X_resampled_test = []
    y_resampled_test = []
    meter_resampled_test = []
    booleans_test    = X_test[:,:] >=  0
    booleans2_test   = X_test[:,:] ==  0
    for j in range(np.shape(X_test)[1]):                                           # Looping over the samples
        label_test = y_test[j]
        count = 0
        
        for i in range(int(np.shape(X_test)[0] / time_points)):  # Looping over the number of chunks assuming no overlap
            # In this statement we take the chunks where all values are greater or equal to zero(I.e excluding the NaN) and where there are at most 25% values equal to zero.
            #Then append them in X_resampled
    
            if  np.all(booleans_test[i*time_points:i*time_points+time_points,j]) == True and np.sum(booleans2_test[i*time_points:i*time_points+time_points,j]) < time_points*zeros_slice_percentage: 
                if np.var(X_test[i*time_points:i*time_points+time_points,j]) > 1e-3:  # Avoid having all the same inputs, want to capture some patterns

                    X_resampled_test.append(X_test[i*time_points:i*time_points+time_points,j])
                    count = count+1
                    y_resampled_test.append(label_test)
                    meter_resampled_test.append(j)
        number_of_chunks_per_customer_test.append(count)



    #Next we build the matrix of slices

    X_resampled = np.stack(X_resampled,axis=-1)  # Stacked data
    X_resampled_test = np.stack(X_resampled_test,axis=-1)  # Stacked data

    #Putting together in one big matrix so it is easier to shuffle.  
    X_big      = np.concatenate([[y_resampled], X_resampled]           ,axis=0)
    X_big      = np.concatenate([[meter_resampled], X_big]             ,axis=0)
    X_big_test = np.concatenate([[y_resampled_test], X_resampled_test] ,axis=0)
    X_big_test = np.concatenate([[meter_resampled_test], X_big_test]   ,axis=0)
                                    
    # Shuffle X_big
    X_big = shuffle_data(X_big)                                    
  
    X_big_test_temp = np.zeros(np.shape(X_big_test))
    
    X_big_test_temp[:,:] = shuffle_data(X_big_test[:,:])
    
    
    X_big_test_temp[2:np.shape(X_big_test_temp)[0],:] = \
        minmax_scale(X_big_test[2:np.shape(X_big_test_temp)[0],:],feature_range=(0,1),axis=0)
    
        
        
    filename_test_time_series ="D:\Master_Thesis_Data\Test_Time_Series" + str(time_points)
    if os.path.exists(filename_test_time_series + ".csv"):
        print("Removing old "+ filename_test_time_series + ".csv"+ " before writing new.")
        os.remove(filename_test_time_series + ".csv")
    np.savetxt(filename_test_time_series +".csv", X_big_test_temp,delimiter = ';')

    # clears some memory
    del(X_big_test_temp)
    
    # Create Loop for taking max number of samples from each meter
                                        
    X_resampled      = []
    X_resampled_test = []
    count_meter      = np.zeros(np.shape(X)[1]) # Number of meters in training set
    count_meter_test = np.zeros(np.shape(X_test)[1])


    for i in range(np.shape(X_big)[1]):     # Go through every sample
        
        index = int(X_big[0,i])             # Converts Float to integer for indexing
        
        if count_meter[index] < number_of_training_slices: # checking if we have no more than max_number of slices of meter index.
            
            X_resampled.append(X_big[:,i])

            count_meter[index] = count_meter[index] + 1 # Keeps track of number of samples extracted from each meter

            # DO THE SAME BUT FOR TEST SET
    for i in range(np.shape(X_big_test)[1]):
        
        index = int(X_big_test[0,i])
        
        if count_meter_test[index] < number_of_test_slices:
            X_resampled_test.append(X_big_test[:,i])
            count_meter_test[index] = count_meter_test[index] + 1

    del(X_big)
    del(X_big_test)                                       
    X_big_tmp      = np.stack(X_resampled,      axis=-1)      
    X_big_test_tmp = np.stack(X_resampled_test, axis=-1)


                                        
    plt.hist(count_meter)
    plt.title("Distribution of number of samples per meter in the training set")
    plt.show()
                                        
    # To keep rest of code running we remove first row so we are left with data containg the label and the data.
    X_big      = np.zeros([np.shape(X_big_tmp)[0]-1,np.shape(X_big_tmp)[1]])
    X_big_test = np.zeros([np.shape(X_big_test_tmp)[0]-1,np.shape(X_big_test_tmp)[1]])


    X_big[:,:]      = X_big_tmp[1:np.shape(X_big_tmp)[0],:]
    X_big_test[:,:] = X_big_test_tmp[1:np.shape(X_big_test_tmp)[0],:]                                  
    del(X_big_tmp)
    del(X_big_test_tmp)
    print(np.shape(X_big))
    print(np.shape(X_big_test))

    # Sorting the datamatrices in order to take same number of samples from each class
    X_big = X_big[:,np.argsort(X_big[0,:])]
    print(X_big)
    X_big_test = X_big_test[:,np.argsort(X_big_test[0,:])]

    a,return_index,return_counts = np.unique(X_big[0,:], return_index=True, return_counts=True)

    print("There are a total of " +str(np.shape(X_big)[1]) +" samples distributed as follows.")
    print(return_counts)
    print(return_index)

    print()
    print()
    a_test,return_index_test,return_counts_test = np.unique(X_big_test[0,:] ,return_index=True, return_counts=True)
    print("The number of samples in each class in test set is distributed as follows" )
    print(return_counts_test)
    print(return_index_test)
    print()

    #Building final dataset of min_number_of_samples per class, randomly sampled and shuffeled
    min_number_of_samples      = np.min(return_counts) # The number of samples from the smallest class
    min_number_of_samples_test = np.min(return_counts_test)

    indexvector = np.arange(0,np.shape(X_big)[1]) # Indices of all samples
    indexvector_test = np.arange(0,np.shape(X_big_test)[1])
    
    data_shuffeled = np.ones([time_points+1,number_of_classes*min_number_of_samples]) #Preallocate memory for final data array.
    data_shuffeled_test = np.ones([time_points+1,number_of_classes*min_number_of_samples_test])


    for i in range(number_of_classes):
    #Picks min_number_of_samples random indices from each class and puts them in the final data array.
        indexchoice = np.random.choice(indexvector[return_index[i]:return_index[i] + return_counts[i]],\
                                       min_number_of_samples,replace=False) 

        indexchoice_test = np.random.choice(indexvector_test[return_index_test[i]:return_index_test[i]+return_counts_test[i]],\
                                            min_number_of_samples_test,replace=False)

        data_shuffeled[:,i*min_number_of_samples:(i+1)*min_number_of_samples] = \
        X_big[:,indexchoice]

        data_shuffeled_test[:,i*min_number_of_samples_test:i*min_number_of_samples_test+min_number_of_samples_test] = \
        X_big_test[:,indexchoice_test]

    print( np.any(np.isnan(data_shuffeled)))
    print(np.any(np.isnan(data_shuffeled_test)))
    # For checking that this works
    a2,return_index2,return_counts2 = np.unique(data_shuffeled[0,:], return_index=True, return_counts=True)  
    a2_test,return_index2_test,return_counts2_test = np.unique(data_shuffeled_test[0,:],return_index = True, return_counts = True)

    print("The number of samples in each class is " +str(min_number_of_samples)+".")
    print(return_counts2)
    print(return_index2)
    print()
    print()

    print("The number of samples in each class for Test set is " +str(min_number_of_samples_test)+".")
    print(return_counts2_test)
    print(return_index2_test)
    print()
    print()

    # Normalization
    
    data_shuffeled[1:np.shape(data_shuffeled)[0],:] = \
        minmax_scale(data_shuffeled[1:np.shape(data_shuffeled)[0],:],feature_range = (0,1),axis = 0,copy=False)
    
    data_shuffeled_test[1:np.shape(data_shuffeled_test)[0],:] = \
        minmax_scale(data_shuffeled_test[1:np.shape(data_shuffeled)[0],:],feature_range=(0,1),axis=0,copy=False)
    
    print("Check for NaN's after feature scaling")
    print(np.any(np.isnan(data_shuffeled)))
    print(np.any(np.isnan(data_shuffeled_test)))
    # Final shuffle of the training data
   
    data_shuffeled = shuffle_data(data_shuffeled)
    # Extract final data, write to csv?
    y_final = data_shuffeled[0,:]
    X_final  = data_shuffeled[1:np.shape(data_shuffeled)[0],:]

    y_one_hot = np.zeros([number_of_classes,np.shape(data_shuffeled)[1]])
    for k in range(np.shape(data_shuffeled)[1]):
        for p in range(number_of_classes):
            if p+1 == y_final[k]:
                y_one_hot[p,k] = 1
    print(np.shape(y_one_hot)[1])

    y_one_hot_test = np.zeros([number_of_classes,np.shape(data_shuffeled_test)[1]])
    for k in range(np.shape(data_shuffeled_test)[1]):
        for p in range(number_of_classes):
            if p+1 == data_shuffeled_test[0,k]:
                y_one_hot_test[p,k] = 1


    # Write processed data into a file.
    filename = "D:\Master_Thesis_Data\Final_Data" + str(time_points)
    if os.path.exists(filename + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
        print("Removing old "+ filename + ".csv"+ " before writing new.")
        os.remove(filename + ".csv")
    np.savetxt(filename + ".csv", data_shuffeled ,delimiter = ';') # Prints data to file
    print("Final_Data Length is "+ str(np.shape(data_shuffeled)[1]))

    filename_y_one_hot = "D:\Master_Thesis_Data\Y_One_Hot" + str(time_points)
    if os.path.exists(filename_y_one_hot + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
        print("Removing old "+ filename_y_one_hot + ".csv"+ " before writing new.")
        os.remove(filename_y_one_hot + ".csv")
    np.savetxt(filename_y_one_hot + ".csv", y_one_hot ,delimiter = ';') # Prints data to file

    print("Final Test Data Length is "+ str(np.shape(data_shuffeled_test)[1]))
    filename_test = "D:\Master_Thesis_Data\Test_Data"+ str(time_points)
    if os.path.exists(filename_test + ".csv"):
        print("Removing old "+ filename_test + ".csv"+ " before writing new.")
        os.remove(filename_test + ".csv")
    np.savetxt(filename_test+".csv", data_shuffeled_test,delimiter = ';')

    filename_y_one_hot_test = "D:\Master_Thesis_Data\Y_One_Hot_Test"+ str(time_points)
    if os.path.exists(filename_y_one_hot_test + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
        print("Removing old "+ filename_y_one_hot_test + ".csv"+ " before writing new.")
        os.remove(filename_y_one_hot_test + ".csv")
    np.savetxt(filename_y_one_hot_test + ".csv", y_one_hot_test ,delimiter = ';') # Prints data to file




    filename_left_out_slice = "D:\Master_Thesis_Data\Left_Out_Data"
    if os.path.exists(filename_left_out_slice + ".csv"):
        print("Removing old "+ filename_left_out_slice + ".csv"+ " before writing new.")
        os.remove(filename_left_out_slice + ".csv")
    np.savetxt(filename_left_out_slice +".csv", np.transpose(left_out_slice),delimiter = ';')
    return 0



for time_points in [24,30,48]:   
    main(time_points)