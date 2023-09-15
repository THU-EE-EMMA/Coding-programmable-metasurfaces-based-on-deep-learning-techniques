# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io as sio
import random
import input_data
import tf_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.00005
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,43200, 0.8, staircase=True)

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
y_predict,rmse,mean_db,f_obj= tf_model.model()
train_step = tf.train.AdamOptimizer(learning_rate).minimize(f_obj,global_step=global_step)
saver=tf.train.Saver(max_to_keep=3)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logs', sess.graph)

sess.run(tf.global_variables_initializer())
x_train,d_train,k_train,y_train = input_data.input_data(test=False)
x_test,d_test,k_test,y_test = input_data.input_data(test=True)

epochs = 210
train_size = x_train.shape[0]
global batch
batch = 40
test_size = x_test.shape[0]

train_index = list(range(x_train.shape[0]))
test_index = list(range(x_test.shape[0]))
find_handle=open('train record.txt',mode='a')
find_handle.write('===============================================================\n')
for i in range(epochs):

    random.shuffle(train_index)
    random.shuffle(test_index)
    x_train,d_train,k_train,y_train = x_train[train_index],d_train[train_index],k_train[train_index],y_train[train_index]
    x_test,d_test,k_test,y_test = x_test[test_index],d_test[test_index],k_test[test_index],y_test[test_index]

    for j in range(0,train_size,batch):
        train_step.run(feed_dict={tf_model.x:x_train[j:j+batch],tf_model.d:d_train[j:j+batch],tf_model.k:k_train[j:j+batch],tf_model.y_actual:y_train[j:j+batch],tf_model.keep_prob:0.5})
    
    temp_trainloss = 0
    train_loss=0
    temp_traindb = 0
    train_db=0
    for j in range(0,train_size,batch):
        train_loss = rmse.eval(feed_dict={tf_model.x:x_train[j:j+batch],tf_model.d:d_train[j:j+batch],tf_model.k:k_train[j:j+batch],tf_model.y_actual:y_train[j:j+batch],tf_model.keep_prob: 1.0})
        train_meandb = mean_db.eval(feed_dict={tf_model.x:x_train[j:j+batch],tf_model.d:d_train[j:j+batch],tf_model.k:k_train[j:j+batch],tf_model.y_actual:y_train[j:j+batch],tf_model.keep_prob: 1.0})
        temp_traindb = temp_traindb+train_meandb
        temp_trainloss = temp_trainloss + train_loss
    train_loss = temp_trainloss/(train_size/batch)
    train_db = temp_traindb/(train_size/batch)

    temp_loss = 0
    temp_grad_x = 0
    temp_db = 0
    temp_fobj = 0
  
    fobj = 0
    meandb = 0
    loss = 0
    for j in range(0,test_size,batch):
        loss = rmse.eval(feed_dict={tf_model.x:x_test[j:j+batch],tf_model.d:d_test[j:j+batch],tf_model.k:k_test[j:j+batch],tf_model.y_actual:y_test[j:j+batch],tf_model.keep_prob: 1.0})
        meandb = mean_db.eval(feed_dict={tf_model.x:x_test[j:j+batch],tf_model.d:d_test[j:j+batch],tf_model.k:k_test[j:j+batch],tf_model.y_actual:y_test[j:j+batch],tf_model.keep_prob: 1.0})
        fobj = f_obj.eval(feed_dict={tf_model.x:x_test[j:j+batch],tf_model.d:d_test[j:j+batch],tf_model.k:k_test[j:j+batch],tf_model.y_actual:y_test[j:j+batch],tf_model.keep_prob: 1.0})
        temp_loss = temp_loss+loss
        temp_db = temp_db+meandb
        temp_fobj = temp_fobj+fobj

    loss = temp_loss/(test_size/batch)
    meandb = temp_db/(test_size/batch)
    fobj = temp_fobj/(test_size/batch)

    if i==100:
        for j in range(0,test_size,batch):
            y_print = y_predict.eval(feed_dict={tf_model.x:x_test[j:j+batch],tf_model.d:d_test[j:j+batch],
                                                tf_model.k:k_test[j:j+batch],tf_model.y_actual:y_test[j:j+batch],
                                                tf_model.keep_prob: 1.0})
            sio.savemat('./100test/'+'100testall'+str(j)+'.mat',{'aa':y_test[j:j+batch],'aa_test':y_print})
        for j in range(0,train_size,batch):
            y_print = y_predict.eval(feed_dict={tf_model.x:x_train[j:j+batch],tf_model.d:d_train[j:j+batch],
                                                tf_model.k:k_train[j:j+batch],tf_model.y_actual:y_train[j:j+batch],
                                                tf_model.keep_prob: 1.0})
            sio.savemat('./100train/'+'100trainall'+str(j)+'.mat',{'aa':y_train[j:j+batch],'aa_test':y_print})
	
    summary_str = sess.run(merged_summary_op,feed_dict={tf_model.x:x_test[0:batch],tf_model.d:d_test[0:batch],
                                                        tf_model.k:k_test[0:batch],tf_model.y_actual:y_test[0:batch],
                                                        tf_model.keep_prob: 1.0})
    summary_writer.add_summary(summary_str,j)
    saver.save(sess, './checkpoint_dir/MyModel', global_step=i)
    print ('epoch {0} done! train_loss:{1} test_loss:{2} f_obj:{3} '
           'traindb:{4} testdb:{5} global_step:{6} learning rate:{7}'.format(i,train_loss,
                                                                             loss,fobj,train_db,meandb,
                                                                             global_step.eval(),learning_rate.eval()))
    find_handle.write('epoch {0} done! train_loss:{1} test_loss:{2} '
                      'f_obj:{3} traindb:{4} testdb:{5} global_step:{6} learning rate:{7}'.format(i,train_loss,
                                                                                                  loss,fobj,train_db,
                                                                                                  meandb,global_step.eval(),
                                                                                                  learning_rate.eval()))
find_handle.close()

