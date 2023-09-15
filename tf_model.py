# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from func import *
import math
with tf.name_scope('input_data') as scope:
    x = tf.placeholder("float",shape=[None,200,200,1],name='input')
    d = tf.placeholder("float",shape=[None,200,200,1],name='input')
    k = tf.placeholder("float",shape=[None,200,200,1],name='input')
    y_actual = tf.placeholder("float",shape=[None,48,48,1],name='results')
    keep_prob = tf.placeholder("float",name='drop_out')

    log10_y=y_actual+2
    batch=40

def model():

    lambda1=0.00002

    tempx=tf.concat(axis=3,values=[d,x,k])
    with tf.name_scope('layer1') as scope:
        W_conv1 = weight_variable([5,5,3,32],lambda1,name='w_conv1')
        b_conv1 = bias_variable([32])
        h_conv1 = conv2d_V1(tempx,W_conv1) + b_conv1
        l1 = max_poolv(tf.nn.relu(h_conv1))

    with tf.name_scope('layer2') as scope:
        W_conv2 = weight_variable([3,3,32,32],lambda1,name='w_conv2')
        b_conv2 = bias_variable([32])
        h_conv2 = conv2d_V1(l1,W_conv2) + b_conv2
        l2 = max_poolv(tf.nn.relu(h_conv2))

    with tf.name_scope('layer3') as scope:
        W_conv3 = weight_variable([3,3,32,64],lambda1,name='w_conv3')
        b_conv3 = bias_variable([64])
        h_conv3 = conv2d_V1(l2,W_conv3) + b_conv3
        l3 = tf.nn.relu(h_conv3)

    with tf.name_scope('layer4') as scope:
        W_conv4 = weight_variable([3,3,64,64],lambda1,name='w_conv4')
        b_conv4 = bias_variable([64])
        h_conv4 = conv2d_V1(l3,W_conv4) + b_conv4
        l4 = max_poolv(tf.nn.relu(h_conv4))

    with tf.name_scope('layer5') as scope:
        W_conv5 = weight_variable([3,3,64,128],lambda1,name='w_conv5')
        b_conv5 = bias_variable([128])
        h_conv5 = conv2d_V1(l4,W_conv5) + b_conv5
        l5 = max_poolv(tf.nn.relu(h_conv5))

    with tf.name_scope('layer6') as scope:
        W_conv6 = weight_variable([3,3,128,256],lambda1,name='w_conv6')
        b_conv6 = bias_variable([256])
        h_conv6 = conv2d_V1(l5,W_conv6) + b_conv6
        l6 = max_poolv(tf.nn.relu(h_conv6))

    with tf.name_scope('layer7') as scope:
        W_conv7 = weight_variable([4,4,256,1600],lambda1,name='w_conv7')
        b_conv7=bias_variable([1600])
        h_conv7=conv2d_V1(l6,W_conv7)+b_conv7
        l7=tf.nn.relu(h_conv7)

    with tf.name_scope('layer8') as scope:
        W_conv71 = weight_variable([1,1,1600,3072],lambda1,name='w_conv8')
        b_conv71 =bias_variable([3072])
        h_conv71 =conv2d_V1(l7,W_conv71)+b_conv71
        l71=tf.nn.relu(h_conv71)

    with tf.name_scope('layer9') as scope:
        W_conv8=weight_variable([1,1,3072,3072],lambda1,name='w_conv9')
        b_conv8=bias_variable([3072])
        h_conv8=conv2d_S(l71,W_conv8)+b_conv8
        l8=tf.nn.relu(h_conv8)+l71

    with tf.name_scope('layer10') as scope:
        W_conv9=weight_variable([1,1,3072,3072],lambda1,name='w_conv10')
        b_conv9=bias_variable([3072])
        h_conv9=conv2d_V1(l8,W_conv9)+b_conv9
        l9=tf.nn.relu(h_conv9)+l71

    with tf.name_scope('layer11') as scope:
        W_conv10=weight_variable([1,1,3072,2304],lambda1,name='w_conv11')
        b_conv10=bias_variable([2304])
        h_conv10=conv2d_V1(l9,W_conv10)+b_conv10
        l10=h_conv10

    y_predict=tf.reshape(l10,[-1,48,48,1])
    y_predict=tf.sigmoid((y_predict-1))+2

    with tf.name_scope('eval_error'):
        with tf.name_scope('rmse') as scope:
            rmse = tf.reduce_mean(tf.div(tf.reduce_sum(tf.square(log10_y - y_predict),[1,2]),tf.reduce_sum(tf.square(log10_y),[1,2])))
            f_obj = rmse
        with tf.name_scope('db') as scope:
            correct_ratio=tf.reduce_sum(tf.cast(tf.equal(tf.round(y_predict),log10_y),tf.float32))/batch/48/48
            mean_db = tf.reduce_mean(correct_ratio)

    db_summary = tf.summary.scalar('db',mean_db)

    return y_predict,rmse,mean_db,f_obj
