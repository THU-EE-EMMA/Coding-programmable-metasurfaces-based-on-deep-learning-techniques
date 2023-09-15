import tensorflow as tf

def weight_variable(shape,lambdareg,name):
    initial = tf.get_variable(name,shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lambdareg)(initial))
    return initial

def weight_variableFCN(shape,lambdareg,name):
    initial = tf.get_variable(name,shape,initializer=tf.contrib.layers.variance_scaling_initializer())
    tf.add_to_collection("losses",tf.contrib.layers.l1_regularizer(lambdareg)(initial))
    return initial

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d_S(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv2d_S2(x,W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')

def conv2d_S4(x,W):
    return tf.nn.conv2d(x,W,strides=[1,4,4,1],padding='SAME')

def conv2d_V1(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def conv2d_V2(x,W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='VALID')

def dconv2d_S2(x,W,outshape):
    return tf.nn.conv2d_transpose(x,W,outshape,strides=[1,2,2,1],padding='SAME')

def dconv2d_S4(x,W,outshape):
    return tf.nn.conv2d_transpose(x,W,outshape,strides=[1,4,4,1],padding='SAME')

def dconv2d_V1(x,W,outshape):
    return tf.nn.conv2d_transpose(x,W,outshape,strides=[1,1,1,1],padding='VALID')

def max_poolv(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

def norm(featuremaps):
    return tf.nn.lrn(featuremaps, 4, bias=1, alpha=0.0001, beta=0.75)

def batch_norm(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, 0.001)

