from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from six.moves import range, zip
import numpy as np
import zhusuan as zs
import pandas as pd
import matplotlib.pyplot as plt

    ## WILL BE BAYESIAN_NN not for now
print("TF Version:", tf.__version__)


def load_data(slice_size):
  # This function loads all of the data that is used for training, takes slice_size as parameter

    ext = ".csv"
    path = "D:\Master_Thesis_Data\Final_Data" + str(slice_size) + ext
    path_y_hot = "D:\Master_Thesis_Data\Y_One_Hot"+ str(slice_size) + ext
    path_test_data = "D:\Master_Thesis_Data\Test_Data"+ str(slice_size) + ext
    path_y_hot_test = "D:\Master_Thesis_Data\Y_One_Hot_Test"+ str(slice_size) + ext
    path_test_time_series = "D:\Master_Thesis_Data\Test_Time_Series"+ str(slice_size) + ext
    # Loading Training Data
    df = pd.read_csv(path,sep=';',header=None)
    data = df.values
    df_y_hot = pd.read_csv(path_y_hot,sep=';',header=None)
    y_one_hot = df_y_hot.values
    number_of_classes = np.shape(y_one_hot)[0]




    # Loading Test Data
    df_test_data = pd.read_csv(path_test_data,sep=';',header=None)
    test_data    = df_test_data.values
    Num_Test_Customers = np.shape(test_data)[1]
    shape_test = [np.shape(test_data)[0],Num_Test_Customers]                    # Test data shapes
    df_y_hot_test = pd.read_csv(path_y_hot_test,sep=';',header=None)
    y_one_hot_test = df_y_hot_test.values
    test_data_X    = test_data[2:np.shape(test_data)[0],:] # Extract time series
    test_data_Y    = test_data[1,:]                        # Extract Labels
    test_data_meter = test_data[0,:]


   
    number_of_meters = int(np.max(np.unique(test_data_meter)) ) # METER NUMBER RANGE
    print(" There are " +str(number_of_meters) + " unique meters in the test dataset.")

 

    y_data = data[1,:]
    X_data  = np.asarray(data[2:np.shape(data)[0],:])

    print( np.any(np.isnan(data)))
    x_train =  np.transpose(X_data[:,:])
    y_train = np.transpose(y_one_hot)
    x_val   = np.transpose(test_data_X[:,:])
    y_data_confusion = test_data_Y
    y_val   =  np.transpose(y_one_hot_test)

    return y_train, x_train, x_val, y_val, y_data_confusion,test_data_meter, \
        number_of_classes,number_of_meters

def main(slice_size,l):
    tf.set_random_seed(1234)
    np.random.seed(1234)
    print("Slice_Size: ", str(slice_size))
    ############ LOADING DATA ##################################
    y_train, X_train, X_val, y_val, y_data_confusion,test_data_meter, \
        number_of_classes,number_of_meters = load_data(slice_size)
    ########################################################## NETWORK #############################################
      
    # This code implements A HIDDEN LAYER NEURAL NETWORK With seven fully connected layers,
    # dropout and batch normalization

    # Learning parameters
    learning_rate = 0.001
    zero_factor = 0.3
    batchsize = 50
    test_one_meter = True 

    # Number of hidden nodes in the network
    N1 = 100
    N2 = 100
    N3 = 100
    

    num_samples = np.shape(X_train)[0]
    input_size  = np.shape(X_train)[1]
    num_classes = number_of_classes
    Nbatches    = int(num_samples/batchsize)
    eps     = 1e-5



    print("Amount of training data is: "+str(np.shape(X_train)[0]))
    print("Amount of validation data is: "+str(np.shape(X_val)[0]))
    # Placeholder for datavectors
    X = tf.placeholder(tf.float32, shape = (None,input_size)) 
    
    y = tf.placeholder(tf.int32, shape = (None,number_of_classes))
    dropout = tf.placeholder_with_default(1.0, shape=())
    istraining = tf.placeholder(tf.bool, shape=[])
    learning_rate_ph = tf.placeholder(tf.float32, shape=())

    initializer = tf.contrib.layers.xavier_initializer()

    ###################################################################################################################
    ########################################### Define the network#####################################################
    ###################################################################################################################
    def mlp(X, weights, biases,dropout,istraining):
        with tf.name_scope("Layer_1"):
            #Layer One
           # fc1 = tf.nn.bias_add(tf.matmul(X,weights['wh1']),biases['b1'])
            fc1 = tf.matmul(X,weights['wh1'])
            fc1 = tf.layers.batch_normalization(fc1,training = istraining)
           # batch_mean1,batch_variance1 = tf.nn.moments(fc1,[0])
           # fc1 = tf.nn.batch_normalization(fc1,batch_mean1,batch_variance1,biases['beta1'],biases['scale1'],epsilon) 
            #fc1 = tf.contrib.layers.batch_norm(fc1,is_training=istraining)
            #fc1 = tf.layers.batch_normalization(fc1, training=istraining)
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1,dropout)
        with tf.name_scope("Layer_2"):
            # Layer Two
            #fc2 = tf.nn.bias_add(tf.matmul(fc1,weights['wh2']),biases['b2'])
            fc2 = tf.matmul(fc1,weights['wh2'])
           # batch_mean2,batch_variance2 = tf.nn.moments(fc2,[0])
            #fc2 = tf.nn.batch_normalization(fc2,batch_mean2,batch_variance2,biases['beta2'],biases['scale2'],epsilon) 
            #fc2 = tf.contrib.layers.batch_norm(fc2,is_training=istraining)
            fc2 = tf.layers.batch_normalization(fc2, training=istraining)
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.nn.dropout(fc2,dropout)
        with tf.name_scope("Layer_3"):
            # Layer Three
          #  fc3 = tf.nn.bias_add(tf.matmul(fc2,weights['wh3']),biases['b3'])
            fc3 = tf.matmul(fc2,weights['wh3'])
          #  batch_mean3,batch_variance3 = tf.nn.moments(fc3,[0])
           # fc3 = tf.nn.batch_normalization(fc3,batch_mean3,batch_variance3,biases['beta3'],biases['scale3'],epsilon) 
           # fc3 = tf.contrib.layers.batch_norm(fc3,is_training=istraining)
            fc3 = tf.layers.batch_normalization(fc3, training=istraining)
            fc3 = tf.nn.relu(fc3)
            fc3 = tf.nn.dropout(fc3,dropout)
     

        with tf.name_scope("Output_Layer"):
            # Return outputs
            pred = tf.nn.bias_add(tf.matmul(fc3,weights['out']),biases['biout'])
            #pred  =tf.nn.relu(pred)

        return pred

    ################################### Set weights and biases ###################################
    weights = {
        # First  hidden layer
        'wh1': tf.Variable(initializer([input_size,N1])),
        # Second hidden layer
        'wh2': tf.Variable(initializer([N1,N2])),
        # Third hidden layer
        'wh3': tf.Variable(initializer([N2,N3])),
       
        # Output layer
        'out': tf.Variable(initializer([N3,num_classes]))
    }

    biases = {
        'scale1': tf.Variable(initializer([N1])),
        'b1' : tf.Variable(initializer([N1])),

        'scale2': tf.Variable(initializer([N2])),
        'b2' : tf.Variable(initializer([N2])),

        'scale3': tf.Variable(initializer([N3])),
        'b3' : tf.Variable(initializer([N3])),
        'biout': tf.Variable(initializer([num_classes]))
    }


    ################################# Model & Evaluation ###################################
    with tf.name_scope("Logits"):
        
        mlp_model = mlp(X,weights,biases,dropout,istraining) # Feeds data through model defined above
        prediction = tf.nn.softmax(mlp_model)     # Constructs a prediction
        y_pred = tf.argmax(prediction,1,output_type=tf.int32)  #


    #################################  Loss and Optimizer ###################################
    with tf.name_scope("Loss"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                            
        
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = mlp_model, labels = y))
        with tf.control_dependencies(update_ops): 
            optimizer = tf.train.AdamOptimizer(learning_rate_ph,beta1 = 0.9 , beta2= 0.999, epsilon=1e-8)
            train_op = optimizer.minimize(loss_op)
        tf.summary.scalar("Validation_Loss",loss_op)
        
        
        
    ################################### Model Evaluation #####################################
    with tf.name_scope("Model_Eval"):  
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.argmax(y,1,output_type=tf.int32)), tf.float32))
        tf.summary.scalar("Accuracy",accuracy)

    ## Training ##

    init = tf.global_variables_initializer()
    
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    

    #writer = tf.summary.FileWriter("C:\Users\cridn_000\Documents\KTH5\Master Thesis\Logs")
    learning_rate = 0.001
    anneal_lr_freq = 1
    anneal_lr_rate = 0.98
    ########################################## SESSION ################################33

    with tf.Session() as sess:
        t1 = time.time()
        sess.run(init)
        print("Optimization Started")
        for epoch in range(5):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices,:]
            y_train = y_train[indices,:]
            print("Epoch "+str(epoch))

            for i in range(Nbatches):
                batch_X = X_train[i*batchsize:(i+1)*batchsize,:]
                batch_y = y_train[i*batchsize:(i+1)*batchsize,:]
                sess.run(train_op,feed_dict={X: batch_X, y: batch_y, dropout: 0.7, \
                    learning_rate_ph: learning_rate, istraining: True})

            valloss,valacc = sess.run([loss_op,accuracy], \
                feed_dict={X: X_val, y: y_val, dropout: 1,istraining: False})
            
            trainloss,trainacc = sess.run([loss_op,accuracy], \
                feed_dict={X: X_train, y: y_train, dropout: 1, istraining: False})
        
            print("Valdiation Loss = " +"{:.4f}".format(valloss) + ", Validation Accuracy = " + "{:.3f}".format(valacc))
            print("Training Loss = " +"{:.4f}".format(trainloss) + ", Training Accuracy = " + "{:.3f}".format(trainacc))

            val_loss.append(valloss)
            val_acc.append(valacc)
            train_loss.append(trainloss)
            train_acc.append(trainacc)
           


        print("Optimization Finished")
    
       

        ####################################################################################
        ########################## FINAL TEST ACCURACY CHECKING ############################
        ####################################################################################
                        ##################################################
            
       
        
        
        ## Running test inference on the different time-series
        print("Starting To Compute The final test accuracy")
        print("Doing this for a varied number of minimal samples")
        fin_acc = []
        for min_num_of_test_samples in range(1, 1, 1):
            
            correct_test_prediction = []
            num_of_test_samples = 300
            #min_num_of_test_samples = 1
            #min_num_of_test_samples = 5
            # start looping through each separate time series
            
            count_class = np.zeros([ number_of_classes ])
            count_meter = np.zeros([ number_of_meters + 1 ])
            test_label = []
            #predictions = np.zeros([number_of_meterseters ]) # stores predictions for each meter

            # WANT TO TAKE K SAMPLES FROM EACH METER AND CLASSIFY THEM CORRECTLY (DONT BOTHER ABVOUT CLASS FOR NOW)

            ## THERE IS A PROBLEM WITH THIS PART OF THE NETWORK,  WON't WORK
            for k in range(number_of_meters+1):
                feed_data_test =[]
                test_label = []
                preds = []
                count=0
                for i in range(np.shape(X_val)[0]): # Looping through each samples
                    tmp = int(test_data_meter[i])      # Temporary Meter Variable
                    tmp_class = int(y_data_confusion[i])
                    # Then we check if we add the test sample to the prediction of some time series
                    if count_meter[tmp] <= num_of_test_samples and tmp == k:
                        test_label.append(tmp_class)
                        feed_data_test.append(X_val[i,:])
                        count_meter[tmp] = count_meter[tmp] + 1
                        count = count+1

                #print(count)

                # Make predictions for the slices of meter k
                if count >= min_num_of_test_samples:
                    # get return counts for each classes
                    feed_data_big = np.transpose(np.stack(feed_data_test, axis = -1))
                    batch_size_temp = 1
                    iters_temp      = int(np.floor(feed_data_big.shape[0] / float(batch_size_temp)))

                    for t in range(iters_temp):
                        feed_data_test_batch = feed_data_big[t * batch_size_temp:(t + 1) * batch_size_temp,:]
                        
                        pred_temp = sess.run(
                            y_pred,
                            feed_dict={X: feed_data_test_batch, dropout: 1,istraining: False}) + 1
                        
                        preds.append(pred_temp)
                    
                #  preds =  sess.run( y_pred,feed_dict={X: feed_data_big, dropout: 1}) + 1
                    a,return_index,return_counts = np.unique(preds, return_index=True, return_counts=True)
                    final_pred = a[np.argmax(return_counts)]
                    if int(final_pred) == int(test_label[0]):
                        correct_test_prediction.append(1)
                    else:
                        correct_test_prediction.append(0)
            
                ## SELECT ONE METER AND PERFORM CHECK  
        

            
            final_test_accuracy = 100*np.round(np.mean(correct_test_prediction),decimals = 3)
            print("Final Test Accuracy is: "+str(final_test_accuracy) + ". Number of Meters Used in evaluation is "+ str(len(correct_test_prediction)))
            t2 = time.time()
            print("Time-Elapsed is " + str((t2-t1)/60) +" minutes.")
            fin_acc.append(final_test_accuracy)

        if(test_one_meter == True):
            for meter_number in test_data_meter:
                # Perform testing on one meter to compare accuracies
                # pick one meter
                count_class = np.zeros([ number_of_classes ])
                count_meter = np.zeros([ number_of_meters + 1 ])
                #meter_number = 251
                feed_data_test =[]
                test_label = []
                preds = []
                num_of_test_samples = 300
                count=0
                for i in range(np.shape(X_val)[0]): # Looping through each samples
                    tmp = int(test_data_meter[i])      # Temporary Meter Variable
                    tmp_class = int(y_data_confusion[i])
                    # Then we check if we add the test sample to the prediction of some time series
                    if count_meter[tmp] <= num_of_test_samples and tmp == meter_number:
                        test_label.append(tmp_class)
                        feed_data_test.append(X_val[i,:])
                        count_meter[tmp] = count_meter[tmp] + 1
                        count = count+1
                # Feed data conains the slices for meter 2
                inds = np.flip(np.arange(0,len(feed_data_test)))
                av = []
                for num_slices in [1,2,4,5,8,10,20,25,40,50,100,200]:
                    split_inds = np.split(inds,len(feed_data_test)/num_slices)
                    correct_test_prediction = []
                    for k in range(len(split_inds)): # Voting loop
                        
                        preds = []
                        for i in range(num_slices):
                            pred_temp = sess.run(
                                y_pred,
                                feed_dict={X: [feed_data_test[split_inds[k][i]]], dropout: 1,istraining: False}) + 1
                            
                            preds.append(pred_temp)
                        a,return_index,return_counts = np.unique(preds, return_index=True, return_counts=True)
                        final_pred = a[np.argmax(return_counts)]
                        if int(final_pred) == int(test_label[0]):
                            correct_test_prediction.append(1)
                        else:
                            correct_test_prediction.append(0)
                    av.append(np.mean(correct_test_prediction))
                    #print(np.mean(correct_test_prediction))    
                
                plt.plot([1,2,4,5,8,10,20,25,40,50,100,200],av,'r',alpha= 0.4)
                plt.ion()
                plt.show()
                plt.pause(0.001)












    filename = "D:\Master_Thesis_Results\Result"
    if os.path.exists(filename + ".csv"):
        print("Removing old "+ filename + ".csv"+ " before writing new.")
        os.remove(filename_left_out_slice + ".csv")
    plt.figure(1)
    plt.plot(np.arange(1,1,1),fin_acc)
    plt.ylabel(" Voting Accuracy ")
    plt.xlabel(" Number of Slices Per Meter ")
    plt.show()
    plt.figure(2)
    plt.plot(val_loss,'r')
    plt.plot(train_loss,'g')
    plt.show()
    tf.reset_default_graph()

    return 0

l=1 # For plotting
for slice_size in [24]:
    filename = "D:\Master_Thesis_Results\Result"
    if os.path.exists(filename + ".csv"):
        print("Removing old "+ filename + ".csv"+ " before writing new.")
        os.remove(filename_left_out_slice + ".csv")
    
    Main_RUN = main(slice_size,l)
    l += 2
