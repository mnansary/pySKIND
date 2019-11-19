#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from progressbar import ProgressBar
import time
import os
import numpy as np 
from glob import glob
from sklearn.model_selection import train_test_split
from coreLib.utils import DataSet,readJson,LOG_INFO,readh5,saveh5,to_tfrecord
#-----------------------------------------------------Load Config----------------------------------------------------------
config_data=readJson('config.json')

class FLAGS:
    DATA_DIR     = config_data['FLAGS']['DATA_DIR']
    CLASSES      = config_data['FLAGS']['CLASSES']
    OUTPUT_DIR   = config_data['FLAGS']['OUTPUT_DIR']
    IMAGE_DIM    = config_data['FLAGS']['IMAGE_DIM']
    ROT_ANGLES   = config_data['FLAGS']['ROT_ANGLES']
    EVAL_SPLIT   = config_data['FLAGS']['EVAL_SPLIT']
    TEST_SPLIT   = config_data['FLAGS']['TEST_SPLIT']
    BATCH_SIZE   = config_data['FLAGS']['BATCH_SIZE']
    
#-----------------------------------------------------------------------------------------------------------------------------------
def create_dataset(FLAGS):
    DS=DataSet(FLAGS)
    DS.create()
    return DS

def get_batched_len(batch_size,data_len):
    new_len= (data_len//batch_size)*batch_size
    return new_len

def create_split(DS,FLAGS):
    LOG_INFO('Data Split')
    # paths for saving data
    X_TEST_PATH     = os.path.join(DS.test_dir,'X_test.h5')
    Y_TEST_PATH     = os.path.join(DS.test_dir,'Y_test.h5')
    X_TRAIN_PATH    = os.path.join(DS.train_dir,'X_train.h5')
    Y_TRAIN_PATH    = os.path.join(DS.train_dir,'Y_train.h5')
    X_EVAL_PATH     = os.path.join(DS.train_dir,'X_eval.h5')
    Y_EVAL_PATH     = os.path.join(DS.train_dir,'Y_eval.h5')
    # complete data
    X=readh5(DS.X_PATH)
    Y=readh5(DS.Y_PATH)
    # split data
    X_comb,X_test,Y_comb,Y_test = train_test_split(X,Y,test_size=FLAGS.TEST_SPLIT)
    X_train,X_eval,Y_train,Y_eval = train_test_split(X_comb,Y_comb,test_size=FLAGS.EVAL_SPLIT)
    # batch data
    train_len=get_batched_len(FLAGS.BATCH_SIZE,X_train.shape[0])
    eval_len =get_batched_len(FLAGS.BATCH_SIZE,X_eval.shape[0])
    # finalize data
    X_train=X_train[:train_len]
    Y_train=Y_train[:train_len]
    X_eval =X_eval[:eval_len]
    Y_eval =Y_eval[:eval_len]
    # save data
    # test
    saveh5(X_TEST_PATH,X_test)
    saveh5(Y_TEST_PATH,Y_test)
    # train
    saveh5(X_TRAIN_PATH,X_train)
    saveh5(Y_TRAIN_PATH,Y_train)
    # eval
    saveh5(X_EVAL_PATH,X_eval)
    saveh5(Y_EVAL_PATH,Y_eval)
    
    LOG_INFO('Train Data Size:{}'.format(X_train.shape[0]))
    LOG_INFO('Test Data Size:{}'.format(X_test.shape[0]))
    LOG_INFO('Eval Data Size:{}'.format(X_eval.shape[0]))
    os.remove(DS.X_PATH)
    os.remove(DS.Y_PATH)
    

def create_tfrecords(DS,FLAGS):
    # complete image paths
    images_path_list=glob(os.path.join(DS.image_dir,'*.png'))
    # test split
    total_data_len=len(images_path_list)
    test_len=int(FLAGS.TEST_SPLIT*total_data_len)
    test_image_path_list=images_path_list[:test_len]
    images_path_list=images_path_list[test_len:]
    # eval and train split
    total_data_len=len(images_path_list)
    eval_len=int(FLAGS.EVAL_SPLIT*total_data_len)
    eval_image_path_list=images_path_list[:eval_len]
    eval_image_path_list = eval_image_path_list[:get_batched_len(FLAGS.BATCH_SIZE,len( eval_image_path_list))]
    train_image_path_list=images_path_list[eval_len:]
    train_image_path_list=train_image_path_list[:get_batched_len(FLAGS.BATCH_SIZE,len(train_image_path_list))]
    # tfrecords
    LOG_INFO('Creating TFRECORDS')
    to_tfrecord(train_image_path_list,'Train',DS.ds_dir,FLAGS.CLASSES)
    to_tfrecord( eval_image_path_list,'Eval' ,DS.ds_dir,FLAGS.CLASSES)
    to_tfrecord( test_image_path_list,'Test' ,DS.ds_dir,FLAGS.CLASSES)
    
    
    

#-----------------------------------------------------------------------------------------------------------------------------------
def main(FLAGS):
    start_time=time.time()
    LOG_INFO('Creating Dataset')
    DS=create_dataset(FLAGS)
    create_split(DS,FLAGS)
    create_tfrecords(DS,FLAGS)
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
    
if __name__ == "__main__":
    main(FLAGS)