import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import time
for slice_size in [ 12,24 ,30 ,48 ,72]: 
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
        test_data_X    = test_data[1:np.shape(test_data)[0],:] # Extract time series
        test_data_Y    = test_data[0,:]                        # Extract Labels

        

        # Test data full time series for final evaluation of sugggested method
        df_test_time_series = pd.read_csv(path_test_time_series,sep=';',header=None)
        test_time_series = df_test_time_series.values
        shape_test_time_series = [np.shape(test_time_series)[0],np.shape(test_time_series)[1]] 

        # Split the final evaluation data into meternumber, label and data
        test_data_meter_time_series = test_time_series[0,:]
        test_data_Y_time_series = test_time_series[1,:]
        test_data_X_time_series = test_time_series[2:shape_test_time_series[0],:]
        number_of_meters = int(np.max(np.unique(test_data_meter_time_series)))
        print(" There are " +str(number_of_meters) + " unique meters in the test dataset.")

        y_one_hot_test_time_series = np.zeros([number_of_classes,np.shape(test_time_series)[1]])
        for k in range(np.shape(test_time_series)[1]):
            for p in range(number_of_classes):
                if p+1 == test_data_Y_time_series[k]:
                    y_one_hot_test_time_series[p,k] = 1
        shape_test_time_series = [np.shape(test_time_series)[0],np.shape(test_time_series)[1]]   


        y_data = data[0,:]
        X_data  = np.asarray(data[1:np.shape(data)[0],:])

        print( np.any(np.isnan(data)))
        X_train =  np.transpose(X_data[:,:])
        y_train = np.transpose(y_one_hot[:,:])
        X_val   = np.transpose(test_data_X[:,:])
        y_val   = np.transpose(y_one_hot_test[:,:])
        y_data_confusion = test_data_Y

        return y_train, X_train, X_val, y_val, y_data_confusion, test_data_X_time_series,test_data_Y_time_series, \
            number_of_classes,test_data_meter_time_series,number_of_meters,y_one_hot_test_time_series

    ############ LOADING DATA ##################################
    y_train, X_train, X_val, y_val, y_data_confusion, test_data_X_time_series,test_data_Y_time_series, number_of_classes,test_data_meter_time_series,number_of_meters,y_one_hot_test_time_series = \
        load_data(slice_size)
    ########################################################## NETWORK #############################################
      
    # This code implements A HIDDEN LAYER NEURAL NETWORK With seven fully connected layers,
    # dropout and batch normalization

    # Learning parameters
    learning_rate = 0.001
    zero_factor = 0.3
    batchsize = 16

    # Number of hidden nodes in the network
    N1 = 800
    N2 = 800
    N3 = 800
    N4 = 800
    N5 = 800
    N6 = 800
    N7 = 800

    num_samples = np.shape(X_train)[0]
    input_size  = np.shape(X_train)[1]
    num_classes = number_of_classes
    Nbatches    = int(num_samples/batchsize)
    epsilon     = 1e-8



    print("Amount of training data is: "+str(np.shape(X_train)[0]))
    print("Amount of validation data is: "+str(np.shape(X_val)[0]))
    # Placeholder for datavectors
    X = tf.placeholder(tf.float32, [None,input_size]) 
    Y = tf.placeholder(tf.float32, [None,num_classes])
    dropout = tf.placeholder_with_default(1.0, shape=())
    initializer = tf.contrib.layers.xavier_initializer()

    ###################################################################################################################
    ########################################### Define the network#####################################################
    ###################################################################################################################
    def mlp(X, weights, biases,dropout):
        with tf.name_scope("Layer_1"):
            #Layer One
            fc1 = tf.matmul(X,weights['wh1'])
            batch_mean1,batch_variance1 = tf.nn.moments(fc1,[0])
            fc1 = tf.nn.batch_normalization(fc1,batch_mean1,batch_variance1,biases['beta1'],biases['scale1'],epsilon) 
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1,dropout)
        with tf.name_scope("Layer_2"):
            # Layer Two
            fc2 = tf.matmul(fc1,weights['wh2'])
            batch_mean2,batch_variance2 = tf.nn.moments(fc2,[0])
            fc2 = tf.nn.batch_normalization(fc2,batch_mean2,batch_variance2,biases['beta2'],biases['scale2'],epsilon) 
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.nn.dropout(fc2,dropout)
        with tf.name_scope("Layer_3"):
            # Layer Three
            fc3 = tf.matmul(fc2,weights['wh3'])
            batch_mean3,batch_variance3 = tf.nn.moments(fc3,[0])
            fc3 = tf.nn.batch_normalization(fc3,batch_mean3,batch_variance3,biases['beta3'],biases['scale3'],epsilon) 
            fc3 = tf.nn.relu(fc3)
            fc3 = tf.nn.dropout(fc3,dropout)
        with tf.name_scope("Layer_4"):
            # Layer Four
            fc4 = tf.matmul(fc3,weights['wh4'])
            batch_mean4,batch_variance4 = tf.nn.moments(fc4,[0])
            fc4 = tf.nn.batch_normalization(fc4,batch_mean4,batch_variance4,biases['beta4'],biases['scale4'],epsilon) 
            fc4 = tf.nn.relu(fc4)
            fc4 = tf.nn.dropout(fc4,dropout)
        with tf.name_scope("Layer_5"):
            # Layer Five
            fc5 = tf.matmul(fc4,weights['wh5'])
            batch_mean5,batch_variance5 = tf.nn.moments(fc5,[0])
            fc5 = tf.nn.batch_normalization(fc5,batch_mean5,batch_variance5,biases['beta5'],biases['scale5'],epsilon) 
            fc5 = tf.nn.relu(fc5)
            fc5 = tf.nn.dropout(fc5,dropout)
        with tf.name_scope("Layer_6"):
            # Layer Six
            fc6 = tf.matmul(fc5,weights['wh6'])
            batch_mean6,batch_variance6 = tf.nn.moments(fc6,[0])
            fc6 = tf.nn.batch_normalization(fc6,batch_mean6,batch_variance6,biases['beta6'],biases['scale6'],epsilon) 
            fc6 = tf.nn.relu(fc6)
            fc6 = tf.nn.dropout(fc6,dropout)

        with tf.name_scope("Layer_7"):
            # Layer 7
            fc7 = tf.matmul(fc6,weights['wh7'])
            batch_mean7,batch_variance7 = tf.nn.moments(fc7,[0])
            fc7 = tf.nn.batch_normalization(fc7,batch_mean7,batch_variance7,biases['beta7'],biases['scale7'],epsilon)
            fc7  =tf.nn.relu(fc7)
            fc7 = tf.nn.dropout(fc7,dropout)

        with tf.name_scope("Output_Layer"):
            # Return outputs
            pred = tf.nn.bias_add(tf.matmul(fc7,weights['out']),biases['biout'])

        return pred

    ################################### Set weights and biases ###################################
    weights = {
        # First  hidden layer
        'wh1': tf.Variable(initializer([input_size,N1])),
        # Second hidden layer
        'wh2': tf.Variable(initializer([N1,N2])),
        # Third hidden layer
        'wh3': tf.Variable(initializer([N2,N3])),
        # Fourth Hidden Layer
        'wh4': tf.Variable(initializer([N3,N4])),
        # Fifth Hidden Layer
        'wh5': tf.Variable(initializer([N4,N5])),
        # Sixth Hidden Layer
        'wh6': tf.Variable(initializer([N5,N6])),
        # Seventh Hidden Layer
        'wh7': tf.Variable(initializer([N6,N7])),
        # Output layer
        'out': tf.Variable(initializer([N7,num_classes]))
    }

    biases = {
        'scale1': tf.Variable(initializer([N1])),
        'beta1' : tf.Variable(initializer([N1])),

        'scale2': tf.Variable(initializer([N2])),
        'beta2' : tf.Variable(initializer([N2])),

        'scale3': tf.Variable(initializer([N3])),
        'beta3' : tf.Variable(initializer([N3])),

        'scale4': tf.Variable(initializer([N4])),
        'beta4' : tf.Variable(initializer([N4])),

        'scale5': tf.Variable(initializer([N5])),
        'beta5' : tf.Variable(initializer([N5])),

        'scale6': tf.Variable(initializer([N6])),
        'beta6' : tf.Variable(initializer([N6])),
        
        'scale7': tf.Variable(initializer([N7])),
        'beta7':  tf.Variable(initializer([N7])),
        'biout': tf.Variable(initializer([num_classes]))
    }


    ################################# Model & Evaluation ###################################
    with tf.name_scope("Logits"):
        mlp_model = mlp(X,weights,biases,dropout) # Feeds data through model defined above
        prediction = tf.nn.softmax(mlp_model)     # Constructs a prediction
        pred_number = tf.argmax(prediction,1)+1   #


    #################################  Loss and Optimizer ###################################
    with tf.name_scope("Loss"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = mlp_model, labels = Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1 = 0.9 , beta2= 0.999, epsilon=1e-8)
        train_op = optimizer.minimize(loss_op)
        tf.summary.scalar("Validation_Loss",loss_op)
        
        
        
    ################################### Model Evaluation #####################################
    with tf.name_scope("Model_Eval"):
        correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        tf.summary.scalar("Accuracy",accuracy)

    ## Training ##

    init = tf.global_variables_initializer()
    tf.set_random_seed(2015)
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []


    #writer = tf.summary.FileWriter("C:\Users\cridn_000\Documents\KTH5\Master Thesis\Logs")

    ########################################## SESSION ################################33

    with tf.Session() as sess:
        t1 = time.time()
        sess.run(init)
        print("Optimization Started")
        for epoch in range(5):
            
            print("Epoch "+str(epoch))

            for i in range(Nbatches):
                batch_X = X_train[i*batchsize:(i+1)*batchsize,:]
                batch_y =y_train[i*batchsize:(i+1)*batchsize,:]
                sess.run(train_op,feed_dict={X: batch_X, Y: batch_y, dropout: 0.6})
            valloss,valacc = sess.run([loss_op,accuracy],feed_dict={X: X_val, Y: y_val, dropout: 1})
            trainloss,trainacc = sess.run([loss_op,accuracy],feed_dict={X: X_train, Y: y_train, dropout: 1})
            
            print("Valdiation Loss = " +"{:.4f}".format(valloss) + ", Validation Accuracy = " + "{:.3f}".format(valacc))
            print("Training Loss = " +"{:.4f}".format(trainloss) + ", Training Accuracy = " + "{:.3f}".format(trainacc))

            val_loss.append(valloss)
            val_acc.append(valacc)
            train_loss.append(trainloss)
            train_acc.append(trainacc)
            # Tensorboard logging
            #validation_log = sess.run(merged,feed_dict={X: X_val, Y: y_val})
            #writer_scalar.add_summary(validation_log,epoch)


        print("Optimization Finished")
    
        predictions = sess.run(pred_number,feed_dict={X: X_val, dropout: 1})
        #print(predictions)
        confusion_matrix = sess.run(tf.confusion_matrix(predictions,y_data_confusion))
        print("Confusion Matrix")
        print(confusion_matrix)

        ####################################################################################
        ########################## FINAL TEST ACCURACY CHECKING ############################
        ####################################################################################
                        ##################################################
            
       
        
        
        print("Starting To Compute The final test accuracy")
        
        correct_test_prediction = []
        num_of_test_samples = 500
        min_num_of_test_samples = 100
        # start looping through each separate time series
       
        count_class = np.zeros([ num_classes ])
        count_meter = np.zeros([ number_of_meters ])
        test_label = []
        #predictions = np.zeros([number_of_meterseters ]) # stores predictions for each meter

        # WANT TO TAKE K SAMPLES FROM EACH METER AND CLASSIFY THEM CORRECTLY (DONT BOTHER ABVOUT CLASS FOR NOW)

        ## THERE IS A PROBLEM WITH THIS PART OF THE NETWORK,  WON't WORK
        for k in range(number_of_meters):
            feed_data_test =[]
            test_label = []
            count=0
            for i in range(np.shape(test_data_X_time_series)[1]): # Looping through each samples
                tmp = test_data_meter_time_series[i]      # Temporary Meter Variable
                tmp_class = test_data_Y_time_series[i]
                # Then we check if we add the test sample to the prediction of some time series
                if count_meter[int(tmp)-1] < num_of_test_samples and tmp-1 == k:
                    test_label.append(tmp_class)
                    feed_data_test.append(test_data_X_time_series[:,i])
                    count_meter[int(tmp)-1] = count_meter[int(tmp)-1] + 1
                    count = count+1
            # Make predictions for the slices of meter k
            if count >= min_num_of_test_samples:
                feed_data_test = np.transpose(np.stack(feed_data_test, axis = -1))
                pred_k = sess.run(pred_number,feed_dict={X: feed_data_test, dropout: 1})
                # get return counts for each classes
                a,return_index,return_counts = np.unique(pred_k, return_index=True, return_counts=True)
                final_pred = a[np.argmax(return_counts)]
                if final_pred == test_label[0]:
                    correct_test_prediction.append(1)
                else:
                    correct_test_prediction.append(0)
        

        #### THIS SHOWS THAT THE NETWORK ACHIEVES OVER 50 % ACCURACY ON THE TESTDATA
        testloss,testacc = sess.run([loss_op,accuracy],feed_dict={X: np.transpose(test_data_X_time_series), Y: np.transpose(y_one_hot_test_time_series), dropout: 1})
        print(testloss)
        print(testacc)
        plt.plot(correct_test_prediction)
        plt.show(block=True)
        final_test_accuracy = np.mean(correct_test_prediction)
        print("Final Accuracy is "+ str(np.round(final_test_accuracy*100,decimals=2))+"%")
        t2 = time.time()
        print("Time-Elapsed is " + str((t2-t1)/60) +" minutes.")

    #writer.add_graph(sess.graph) 
    plt.figure(1)
    plt.plot(val_loss,'r')
    plt.plot(train_loss,'g')
    plt.show(block=True)
