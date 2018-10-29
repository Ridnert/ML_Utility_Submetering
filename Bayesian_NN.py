##

# In this file I will try to implement a bayesian neural network on the same dataset.

# The package used will be ZhuSuan: A Library for Bayesian Deep Learning
# **ZhuSuan** is a python probabilistic programming library for Bayesian deep
# learning, which conjoins the complimentary advantages of Bayesian methods and
# deep learning. ZhuSuan is built upon
# [Tensorflow](https://www.tensorflow.org). Unlike existing deep
# learning libraries, which are mainly designed for deterministic neural
# networks and supervised tasks, ZhuSuan provides deep learning style primitives
# and algorithms for building probabilistic models and applying Bayesian
# inference. 

# Using Tensorflow-GPU 1.13
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import time
import zhusuan as zs

## FUNCTIONS
@zs.reuse('model')
def bayesianNN(observed, x, n_x, weights, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
       # for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                         #       layer_sizes[1:])):
        #    w_mu = tf.zeros([1, n_out, n_in + 1])


        for i in len(weights):

            ws.append(
                zs.Normal('w' + str(i), w_mu, std=1.,
                            n_samples=n_particles, group_ndims=2))

        with tf.name_scope("Layer_1"):
            #Layer One
            fc1 = tf.matmul(x,weights['wh1'])
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


        # forward
        ly_x = tf.expand_dims(
             tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat(
                [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            ly_x = tf.matmul(w, ly_x) / \
                tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        y_mean = tf.squeeze(ly_x, [2, 3])
        y_logstd = tf.get_variable(
            'y_logstd', shape=[],
            initializer=tf.constant_initializer(0.))
        y = zs.Normal('y', y_mean, logstd=y_logstd)

    return model, y_mean


def mean_field_variational(layer_sizes, n_particles):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mean = tf.get_variable(
                'w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable(
                'w_logstd_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            ws.append(
                zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                          n_samples=n_particles, group_ndims=2))
    return variational

def log_joint(observed):
    model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
    log_pws = model.local_log_prob(w_names)
    log_py_xw = model.local_log_prob('y')
    return tf.add_n(log_pws) + log_py_xw * N





##########################################################################################
############################# LOADING DATA ################################################
###########################################################################################
print("TF Version:", tf.__version__)
path = "D:\Master_Thesis_Data\Final_Data.csv"
path_y_hot = "D:\Master_Thesis_Data\Y_One_Hot.csv"
path_test_data = "D:\Master_Thesis_Data\Test_Data.csv"
path_y_hot_test = "D:\Master_Thesis_Data\Y_One_Hot_Test.csv"
path_test_time_series = "D:\Master_Thesis_Data\Test_Time_Series.csv"


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
test_data_X    = test_data[1:np.shape(test_data)[0],1:1000] # Extract time series
test_data_Y    = test_data[0,1:1000]                        # Extract Labels


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

########################################################## NETWORK #############################################

num_samples = np.shape(X_data)[1]
input_size  = np.shape(X_data)[0]
num_classes = number_of_classes

epsilon     = 1e-8

################ Transposing datasets for correct input shape to the network ##################
X_train =  np.transpose(X_data[:,:])
y_train = y_data# np.transpose(y_one_hot[:,:])
X_val   = np.transpose(test_data_X[:,:])
y_val   =  test_data_Y #np.transpose(y_one_hot_test[:,:])
y_data_confusion = test_data_Y
N, n_x = X_train.shape
print("Amount of training data is: "+str(np.shape(X_train)[0]))
print("Amount of validation data is: "+str(np.shape(X_val)[0]))
# Placeholder for datavectors
# Define model parameters
n_hiddens = [50]
n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles') # Number of particles in each hidden layer
x = tf.placeholder(tf.float32, [None,n_x]) 
y = tf.placeholder(tf.float32, [None])
dropout = tf.placeholder_with_default(1.0, shape=())
layer_sizes = [n_x] + n_hiddens + [1]
print(layer_sizes)
print(layer_sizes[:-1])
print(layer_sizes[1:])
w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]
initializer = tf.contrib.layers.xavier_initializer()

###################################################################################################################
########################################### Define the network#####################################################
###################################################################################################################


variational = mean_field_variational(layer_sizes, n_particles)
qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
latent = dict(zip(w_names, qw_outputs))
lower_bound = zs.variational.elbo(log_joint, observed={'y': y}, latent=latent, axis=0)
cost = tf.reduce_mean(lower_bound.sgvb())
lower_bound = tf.reduce_mean(lower_bound)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
infer_op = optimizer.minimize(cost)


 # prediction: rmse & log likelihood
observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
observed.update({'y': y})
model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
y_pred = tf.reduce_mean(y_mean, 0)
correct_pred = tf.equal(tf.round(y_pred),y)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) #* std_y_train
log_py_xw = model.local_log_prob('y')
log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))# - \
   # tf.log(std_y_train)
# Define training/evaluation parameters
lb_samples = 10
ll_samples = 500
epochs = 2
batch_size = 10
iters = int(np.floor(X_train.shape[0] / float(batch_size)))
test_freq = 2
writer = tf.summary.FileWriter(r"C:\Users\cridn_000\Documents\KTH5\Master Thesis\Logs")


N1 = 100
N2 = 100

#### DEFINING THE LAYERS
 weights = {
        # First  hidden layer
        'wh1': tf.Variable(initializer([input_size,N1])),
        # Second hidden layer
        'wh2': tf.Variable(initializer([N1,N2]))
        }



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, epochs + 1):
        lbs = []
        for t in range(iters):
            x_batch = X_train[t*batch_size:(t+1)*batch_size,:]
            y_batch = y_train[t*batch_size:(t+1)*batch_size]
            _, lb = sess.run(
                [infer_op, lower_bound],
                feed_dict={n_particles: lb_samples,  x: x_batch, y: y_batch})
            lbs.append(lb)
        print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))
        ypred = sess.run(y_pred, feed_dict = {n_particles: ll_samples, x: X_val})
        #print(np.equal(np.round(ypred),y_val))
       # print(np.mean(np.equal(np.round(ypred),y_val)))
        #print(ypred)
        if epoch % test_freq == 0:
            test_lb, test_rmse, test_ll,test_acc = sess.run( [lower_bound, rmse, log_likelihood,accuracy],feed_dict={n_particles: ll_samples,  x: X_val, y: y_val})
            print('>> TEST')
            print('>> Test lower bound = {}, rmse = {}, log_likelihood = {}, Accuracy = {}' .format(test_lb, test_rmse, test_ll,test_acc))

writer.add_graph(sess.graph) 