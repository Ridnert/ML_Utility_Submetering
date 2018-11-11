
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
        y_train = y_data
        X_val   = np.transpose(test_data_X[:,:])
        y_val   = test_data_Y
        y_data_confusion = test_data_Y

        return y_train, X_train, X_val, y_val, y_data_confusion, test_data_X_time_series,test_data_Y_time_series, \
            number_of_classes,test_data_meter_time_series,number_of_meters,y_one_hot_test_time_series

    ############ LOADING DATA ##################################
    

@zs.reuse('model')
def var_dropout(observed, x, n, net_size, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        h = x
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
            eps_mean = tf.ones([n, n_in])
            eps = zs.Normal(
                'layer' + str(i) + '/eps', eps_mean, std=1.,
                n_samples=n_particles, group_ndims=1)
            h = layers.fully_connected(
                h * eps, n_out, normalizer_fn=layers.batch_norm,
                normalizer_params=normalizer_params)
            if i < len(net_size) - 2:
                h = tf.nn.relu(h)
        y = zs.Categorical('y', h)
    return model, h


@zs.reuse('variational')
def q(observed, n, net_size, n_particles):
    with zs.BayesianNet(observed=observed) as variational:
        for i, [n_in, n_out] in enumerate(zip(net_size[:-1], net_size[1:])):
            with tf.variable_scope('layer' + str(i)):
                logit_alpha = tf.get_variable('logit_alpha', [n_in])

            alpha = tf.nn.sigmoid(logit_alpha)
            alpha = tf.tile(tf.expand_dims(alpha, 0), [n, 1])
            eps = zs.Normal('layer' + str(i) + '/eps',
                            1., logstd=0.5 * tf.log(alpha + 1e-10),
                            n_samples=n_particles, group_ndims=1)
    return variational


if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load MNIST
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train),(x_test, y_test) = mnist.load_data()
    #x_train, x_test = x_train / 255.0, x_test / 255.0
    
    
    slice_size = 24
    n_x = slice_size
    y_train, x_train, x_test, y_test, y_data_confusion, test_data_X_time_series,test_data_Y_time_series, number_of_classes,test_data_meter_time_series,number_of_meters,y_one_hot_test_time_series = \
        load_data(slice_size)
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    # Define training/evaluation parameters
    epochs = 50
    batch_size = 100
    lb_samples = 10
    ll_samples = 100
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 5
    learning_rate = 0.001
    anneal_lr_freq = 100
    anneal_lr_rate = 0.75

    # placeholders
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.float32, shape=(None, n_x))
    y = tf.placeholder(tf.int32, shape=(None))
    n = tf.shape(x)[0]

    net_size = [n_x, 100, 100, 100, 10]
    e_names = ['layer' + str(i) + '/eps' for i in range(len(net_size) - 1)]

    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    y_obs = tf.tile(tf.expand_dims(y, 0), [n_particles, 1])

    def log_joint(observed):
        model, _ = var_dropout(observed, x_obs, n, net_size,
                               n_particles, is_training)
        log_pe = model.local_log_prob(e_names)
        log_py_xe = model.local_log_prob('y')
        return tf.add_n(log_pe) / x_train.shape[0] + log_py_xe

    variational = q({}, n, net_size, n_particles)
    qe_queries = variational.query(e_names, outputs=True, local_log_prob=True)
    qe_samples, log_qes = zip(*qe_queries)
    log_qes = [log_qe / x_train.shape[0] for log_qe in log_qes]
    e_dict = dict(zip(e_names, zip(qe_samples, log_qes)))
    lower_bound = zs.variational.elbo(log_joint, {'y': y_obs}, e_dict, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    _, h_pred = var_dropout(dict(zip(e_names, qe_samples)),
                            x_obs, n, net_size,
                            n_particles, is_training)
    h_pred = tf.reduce_mean(tf.nn.softmax(h_pred), 0)
    y_pred = tf.argmax(h_pred, 1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y), tf.float32))

    learning_rate_ph = tf.placeholder(tf.float32, shape=())
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
                _, lb = sess.run(
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
                for t in range(10):
                    x_batch = x_test[t * 100:(t + 1) * 100,:]
                    y_batch = y_test[t * 100:(t + 1) * 100]
                    lb, acc1 = sess.run(
                        [lower_bound, acc],
                        feed_dict={n_particles: ll_samples,
                                   is_training: False,
                                   x: x_batch, y: y_batch})
                    test_lbs.append(lb)
                    test_accs.append(acc1)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test accuaracy = {}'.format(np.mean(test_accs)))
        print("DONE!")