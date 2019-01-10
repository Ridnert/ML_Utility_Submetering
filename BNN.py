
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
from matplotlib import pyplot

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
        

        
        max_meter_number = int(np.max(np.unique(test_data_meter))) # METER NUMBER RANGE
        number_of_meters = int(len(np.unique(test_data_meter)))
        print(" There are " +str(number_of_meters) + " unique meters in the test dataset.")
     

        y_data = data[1,:]
        X_data  = np.asarray(data[2:np.shape(data)[0],:])

        print( np.any(np.isnan(data)))
        x_train =  np.transpose(X_data[:,:])
        y_train = y_data
        x_val   = np.transpose(test_data_X[:,:])
        y_val   = test_data_Y
        y_data_confusion = test_data_Y

        return y_train, x_train, x_val, y_val, y_data_confusion,test_data_meter, \
            number_of_classes,number_of_meters,max_meter_number

    ############ LOADING DATA ##################################
    

@zs.reuse('model')
def var_dropout(observed, x, n, net_size, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        h = x
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
            eps_mean = tf.zeros([n, n_in])
            # Adding noise to Weights
            eps = zs.Normal(
                'layer' + str(i) + '/eps', eps_mean, std=1.,
                n_samples=n_particles, group_ndims=1)

            h = layers.fully_connected(
                h * eps, n_out, normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params)
            if i < len(net_size) - 2:
                h = tf.nn.relu(h)

            
        y = zs.Categorical('y', h,group_ndims=0)
        
    return model, h


@zs.reuse('variational')
def q(observed, n, net_size, n_particles):
    # Build the variational posterior distribution.
    # We assume it is factorized
    with zs.BayesianNet(observed=observed) as variational:
        for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
            with tf.variable_scope('layer' + str(i)):
                logit_alpha = tf.get_variable('logit_alpha', [n_in])
            
            alpha = tf.nn.sigmoid(logit_alpha)
            alpha = tf.tile(tf.expand_dims(alpha, 0), [n, 1])
            eps = zs.Normal('layer' + str(i) + '/eps',
                            1., logstd=0.5 * tf.log(alpha + 1e-6),
                            n_samples=n_particles, group_ndims=1)
    return variational


def log_joint(observed):
    # Defines the log joint likelihood of model and variational objectives.
    model, _ = var_dropout(observed, x_obs, n, net_size,
                            n_particles, is_training)
    log_pW = model.local_log_prob(W_names)
    log_py_xW = model.local_log_prob('y')
    return tf.add_n(log_pW)/ x_train.shape[0]  + log_py_xW

#/ x_train.shape[0]

def entropy(p_vec):
    
    e = 0
    for k in range(len(p_vec)):
        if p_vec[k] !=0:
            e = e + p_vec[k]*np.log(p_vec[k]) 
    return -e


if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)

    
    slice_size = 24

    n_x = slice_size
    y_train, x_train, x_test, y_test, y_data_confusion,test_data_meter, \
            number_of_classes,number_of_meters,max_meter_number = \
        load_data(slice_size)
    
    y_train = y_train.astype(int)-1
    
    y_test = y_test.astype(int)-1
    # Define training/evaluation parameters
    epochs = 20
    batch_size = 20
    batch_size_test = 20
    lb_samples = 30
    ll_samples = 500
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    iters_test = int(np.floor(x_test.shape[0] / float(batch_size_test)))
    test_freq = 4
    learning_rate = 0.001
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75
    entropy_checking = False

    # placeholders
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.int32, shape=(None))
    n = tf.shape(x)[0]
    learning_rate_ph = tf.placeholder(tf.float32, shape=())



    net_size = [n_x, 100, 100, 100, number_of_classes]
    W_names = ['layer' + str(i) + '/eps' for i in range(len(net_size) - 1)]

    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])



    variational = q({},n, net_size, n_particles)
    # Get samples and log probabilities from the variational posterior
    # Query stochastic nodes in W_names about output and local log probabilities.
    qW_queries = variational.query(W_names, outputs=True, local_log_prob=True)
    qW_samples, log_qWs = zip(*qW_queries)
    log_qWs = [log_qW / x_train.shape[0] for log_qW in log_qWs]
    W_dict = dict(zip(W_names, zip(qW_samples, log_qWs)))
    # wdict of the form {'W_names': [qW_samples, log_qWs] } input to the ELBO
    lower_bound = zs.variational.elbo(log_joint, {'y': y_obs}, W_dict, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    
    lower_bound_exp = zs.variational.elbo(log_joint,{'y': y_obs},W_dict,axis=0)
    
    # Predictions
    model, h_pred = var_dropout(dict(zip(W_names, qW_samples)),
                            x_obs, n, net_size,
                            n_particles, is_training)
    h_pred = tf.reduce_mean(tf.nn.softmax(h_pred), 0)
    y_pred = tf.argmax(h_pred, 1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y), tf.float32))

    log_py_xw = model.local_log_prob('y')
    log_likelihood = zs.log_mean_exp(log_py_xw, 0)
        


    
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    infer = optimizer.minimize(cost)

    params = tf.trainable_variables()
    for i in params:
        print('variable name = {}, shape = {}'
              .format(i.name, i.get_shape()))

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            time_epoch = -time.time()
            indices = np.random.permutation(x_train.shape[0])
            x_train = x_train[indices,:]
            y_train = y_train[indices]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size,:]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb  = sess.run(
                    [infer, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               is_training: True,
                               learning_rate_ph: learning_rate,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_accs = []
                test_ll = []
                test_preds = []
                test_entropies_correct = []
                test_entropies_wrong   = []
                for t in range(iters_test):
                    x_batch = x_test[t * batch_size_test:(t + 1) * batch_size_test,:]
                    y_batch = y_test[t * batch_size_test:(t + 1) * batch_size_test]
                    lb, acc1,ll,pred_vec,y_preds = sess.run(
                        [lower_bound, acc,log_likelihood,h_pred,y_pred],
                        feed_dict={n_particles: ll_samples,
                                   is_training: False,
                                   x: x_batch, y: y_batch})

                    test_lbs.append(lb)
                    test_accs.append(acc1)
                    test_ll.append(ll)
                    if(entropy_checking == True): 
                        ###########################################################################
                        # Want to collect entropies as functions of correct or wrong predictions  #
                        ###########################################################################
                        for i in range(batch_size_test): 
                            if y_preds[i] == y_batch[i]:
                                test_entropies_correct.append(entropy(pred_vec[i,:]))
                            else:
                                test_entropies_wrong.append(entropy(pred_vec[i,:]))

                    
                if(entropy_checking == True):
                    print(np.mean(test_entropies_correct))
                    print(np.mean(test_entropies_wrong))
                    bins = np.linspace(0.5, 1.5, 100)
                    pyplot.hist(test_entropies_correct, bins, alpha=0.5, label='Correct Predictions')
                    pyplot.hist(test_entropies_wrong, bins, alpha=0.5, label='Wrong Predictions')
                    pyplot.legend(loc='upper right')
                    plt.title("Distribution Of Entropy of Likelihood")
                    pyplot.show()
                time_test += time.time()
               # print(np.mean(pred,axis = 0))
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test accuracy = {},  log(p(y|x,W)) = {} '.format(np.mean(test_accs), (np.mean(test_ll))))


        ## Running test inference on the different time-series
        print("Starting To Compute The final test accuracy")
        fin_acc = []
        num_met =[]
        for gurk in [1,0.1,0.2,0.5,0.8]: #range(1,50,5):    
            correct_test_prediction = []
            num_of_test_samples = 30000
            min_num_of_test_samples = 1
            # start looping through each separate time series
            
            count_class = np.zeros([ number_of_classes ])
            count_meter = np.zeros([ max_meter_number+1 ])
            test_label = []
            #predictions = np.zeros([number_of_meterseters ]) # stores predictions for each meter

            # WANT TO TAKE K SAMPLES FROM EACH METER AND CLASSIFY THEM CORRECTLY (DONT BOTHER ABVOUT CLASS FOR NOW)

            ## THERE IS A PROBLEM WITH THIS PART OF THE NETWORK,  WON't WORK
            for k in range(max_meter_number+1):
                feed_data_test =[]
                test_label = []
                preds = []
                entropies = []
                count=0
                for i in range(np.shape(x_test)[0]): # Looping through each samples
                    tmp = int(test_data_meter[i])      # Temporary Meter Variable
                    tmp_class = y_test[i]
                    # Then we check if we add the test sample to the prediction of some time series
                    if count_meter[int(tmp)] <= num_of_test_samples and tmp == k:
                        test_label.append(tmp_class)
                        feed_data_test.append(x_test[i,:])
                        count_meter[int(tmp)] = count_meter[int(tmp)] + 1
                        count = count + 1



                # Make predictions for the slices of meter k
                if count >= min_num_of_test_samples:
                    # get return counts for each classes
                    feed_data_big = np.transpose(np.stack(feed_data_test, axis = -1))
                    batch_size_temp = 1
                    iters_temp      = int(np.floor(feed_data_big.shape[0] / float(batch_size_temp)))

                    for t in range(iters_temp):
                        feed_data_test_batch = feed_data_big[t * batch_size_temp:(t + 1) * batch_size_temp,:]
                        
                        if(True):
                            ytmp = sess.run(
                                [h_pred],
                                feed_dict={n_particles: ll_samples,
                                        is_training: False,
                                        x: feed_data_test_batch})

                            
                            ytmp = np.squeeze(ytmp)
                            
                            entropies.append(entropy(ytmp))
                            #if entropies[k] < 0.25:
                            pred_temp = np.argmax(ytmp)
                            preds.append(pred_temp)
                        
                        if(False):
                            lb = []
                            
                            xtmp = feed_data_test_batch
                            for k in range(number_of_classes):
                                ys = [k]
                                lb.append(sess.run([lower_bound],feed_dict={n_particles: ll_samples,is_training: False, x: xtmp, y: ys }))
                            pred_temp = np.argmax(lb)
                        
                            preds.append(pred_temp)
                    
                    if len(preds) > 0:
                        
                        indtmp =  np.argsort(entropies)
                      
                        new_preds=[]
                        for i in range(int(np.ceil(len(entropies)*gurk))):
                            new_preds.append(preds[indtmp[i]])
                        
                    a,return_index,return_counts = np.unique(new_preds, return_index=True, return_counts=True)
                    final_pred = a[np.argmax(return_counts)]
                    if int(final_pred) == int(test_label[0]):
                        correct_test_prediction.append(1)
                    else:
                        correct_test_prediction.append(0)
            final_test_accuracy = 100*np.round(np.mean(correct_test_prediction),decimals = 3)
            fin_acc.append(final_test_accuracy)
            num_met.append(len(correct_test_prediction))
            print("Final Test Accuracy is "+str(final_test_accuracy) + "Number of Meters Used in evaluation is "+ str(len(correct_test_prediction)))
        plt.figure(1)
        plt.plot(fin_acc)
        plt.show()
        plt.figure(2)
        plt.plot(num_met)
        plt.show()
        
        

        print("DONE!")