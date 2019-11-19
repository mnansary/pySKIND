# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import os
import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image as imgop
import imageio
from glob import glob
import h5py
import json
import random
import tensorflow as tf 

from progressbar import ProgressBar
#---------------------------------------------------------------------------
def readJson(file_name):
    return json.load(open(file_name))

def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
    hf.close()

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

def LOG_INFO(log_text,p_color='green',rep=True):
    if rep:
        print(colored('#    LOG:','blue')+colored(log_text,p_color))
    else:
        print(colored('#    LOG:','blue')+colored(log_text,p_color),end='\r')

def create_dir(base_dir,ext_name):
    new_dir=os.path.join(base_dir,ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir
#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    '''
    This Class is used to preprocess The dataset for Training 
    One single Image is augmented with rotation of (STATICS.rot_angle_start,STATICS.rot_angle_end) increasing by STATICS.rot_angle_step 
    and each rotated image is flipped horizontally,vertically and combinedly to produce 4*N*rot_angles images per one input
    '''
    def __init__(self,FLAGS):
        self.data_dir    = FLAGS.DATA_DIR
        self.ds_dir      = create_dir(FLAGS.OUTPUT_DIR,'DataSet')
        self.train_dir   = create_dir(self.ds_dir,'Train')
        self.test_dir    = create_dir(self.ds_dir,'Test')
        self.image_dir   = create_dir(self.ds_dir,'Images')
        self.image_dim   = FLAGS.IMAGE_DIM
        self.classes     = str(FLAGS.CLASSES).split(',')
        self.rot_angles  = str(FLAGS.ROT_ANGLES).split(',')
        self.X_PATH      = os.path.join(self.ds_dir,'X.h5')
        self.Y_PATH      = os.path.join(self.ds_dir,'Y.h5')
        self.Xs          =[]
        self.Ys          =[]
        
    
    def __listIMGS(self):
        self.img_dirs=[]
        for iden in self.classes:
            self.img_dirs+=glob(os.path.join(self.data_dir,iden,'*.jpg'))
        random.shuffle(self.img_dirs)

    def __processIMGS(self):
        LOG_INFO('Creating Images',p_color='yellow')
        _pbar=ProgressBar()
        SAMPLE_NO=0
        for img_dir in _pbar(self.img_dirs):
            #base=os.path.basename(img_dir).split('.')[0]
            SAMPLE_NO+=1
            label=os.path.basename(os.path.dirname(img_dir))
            # get image
            img=imgop.open(img_dir)
            img=img.resize((self.image_dim,self.image_dim))
            # rotation loops
            for rot_angle in self.rot_angles:
                rot_img=img.rotate(int(rot_angle))
                for fid in range(4):
                    rand_id=random.randint(0,10E4)
                    x=self.__getFlipDataById(rot_img,fid)
                    file_path=os.path.join(self.image_dir,'{}_{}_{}_{}_{}.png'.format(rand_id,rot_angle,fid,SAMPLE_NO,label))
                    imageio.imsave(file_path,x)
        
    def __getFlipDataById(self,img,fid):
        if fid==0:# ORIGINAL
            x=np.array(img)
        elif fid==1:# Left Right Flip
            x=np.array(img.transpose(imgop.FLIP_LEFT_RIGHT))
        elif fid==2:# Up Down Flip
            x=np.array(img.transpose(imgop.FLIP_TOP_BOTTOM))
        else: # Mirror Flip
            x=img.transpose(imgop.FLIP_TOP_BOTTOM)
            x=np.array(x.transpose(imgop.FLIP_LEFT_RIGHT))
        return x

    def __SaveFeatsLabels(self):
        LOG_INFO('Saving Features and Labels',p_color='yellow')
        label_0='_{}'.format(self.classes[0])
        img_dirs=glob(os.path.join(self.image_dir,'*.png'))
        _pbar=ProgressBar()
        for img_dir in _pbar(img_dirs):
            y=[0 for _ in range(len(self.classes))]
            # get Y:
            if label_0 in img_dir:
                y[0]=1
            else:
                y[1]=1
            y=np.array(y)    
            y=np.expand_dims(y,axis=0)
            # get X
            x=np.array(imgop.open(img_dir))
            tensor=np.expand_dims(x,axis=0)
            self.Xs.append(tensor)
            self.Ys.append(y)
        X=np.vstack(self.Xs)
        Y=np.vstack(self.Ys)
        saveh5(self.X_PATH,X)
        saveh5(self.Y_PATH,Y)

    def create(self):
        self.__listIMGS()
        self.__processIMGS()
        self.__SaveFeatsLabels()
#--------------------------------------------------------------------------------------------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths,mode,data_dir,CLASSES):
    '''
    Creates tfrecords from Provided Image Paths
    Arguments:
    image_paths = List of image paths
    data_dir    = Path to save the tfrecords
    mode        = Mode of data to be created
    rec_num       = number of record
    '''
    # Create Saving Directory based on mode
    save_dir=create_dir(data_dir,'TFRECORD')
    tfrecord_name='{}.tfrecord'.format(mode)
    tfrecord_path=os.path.join(save_dir,tfrecord_name) 
    # class
    classes= str(CLASSES).split(',')
    label_0='_{}'.format(classes[0])
    # writting to tfrecords
    LOG_INFO(mode)
    _bar=ProgressBar()
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in _bar(image_paths):
            #labels
            if label_0 in image_path:
                label=0
            else:
                label=1
            # image data
            with(open(image_path,'rb')) as fid:
                image_png_bytes=fid.read()
            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'label':_int64_feature(label)
            }
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)   
#--------------------------------------------------------------------------------------------------------------------------------------------------
def data_input_fn(FLAGS,params,nb_classes=2): 
    '''
    This Function generates data from provided FLAGS
    FLAGS must include:
        TFRECORDS_PATH  = Path to tfrecords
        IMAGE_DIM       = Dimension of Image
        NB_CHANNELS     = Depth of Image
        BATCH_SIZE      = batch size for traning
        SHUFFLE_BUFFER  = Buffer Size > Batch Size
        MODE            = 'train/eval'
    params          = Needed for estimator to pass batch size
    '''
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'label'  : tf.io.FixedLenFeature([],tf.int64)
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=FLAGS.NB_CHANNELS)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        idx = tf.cast(parsed_example['label'], tf.int32)
        label=tf.one_hot(idx,nb_classes,dtype=tf.int32)
        return image,label

    file_paths=glob(os.path.join(FLAGS.TFRECORDS_DIR,'{}.tfrecord'.format(FLAGS.MODE)))
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(FLAGS.SHUFFLE_BUFFER,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.BATCH_SIZE,drop_remainder=True)
    return dataset
    