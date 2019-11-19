# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation
from tensorflow.keras.layers import Dropout,Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D,Input,Concatenate,BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

import os
#--------------------------------------------------------------------------------------
def conv_net(img_dim=64,nb_channels=1):
    img_shape=(img_dim,img_dim,nb_channels)
    # layer specs
    filters=[64,128,256,512]
    d_layes=[256,512]
    # conv and pool
    model = Sequential()
    # 1st conv
    model.add(Conv2D(64, (2, 2), input_shape=img_shape,activation='relu',padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # conv_layers
    for nb_filter in filters:
        model.add(Conv2D(nb_filter, (2, 2),activation='relu',padding='valid'))
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    # dense_layers
    for d in d_layes:
        model.add(Dense(d,activation='relu'))
    model.add(Dense(2,activation = 'softmax'))
    return model
#--------------------------------------------------------------------------------------
class DenseNet(object):
    """
        Fast Implementation of DenseNet based on: https://github.com/Lasagne/Recipes/blob/master/papers/densenet/densenet_fast.py
        Heavily Borrowed From: https://github.com/titu1994/DenseNet/blob/master/densenet_fast.py
    """
    def __init__(
        self,image_dim=64,
        nb_channels=1,
        nb_classes=2,
        nb_dense=3,
        nb_layers=12,
        nb_filter=12, 
        growth_rate=12, 
        weight_decay=1e-4,
        dropout_rate=None,
        bottleneck=False,
        compression=0.5):
        
        # model params
        self.IN = Input(shape=(image_dim,image_dim,nb_channels),name='MODEL_INPUT')
        self.nb_classes=nb_classes
        # filter count
        self.STARTING_FILTER=nb_filter
        self.nb_filter=nb_filter
        self.growth_rate=growth_rate
        self.compression=compression
        # block params
        self.nb_dense=nb_dense
        self.bottleneck=bottleneck
        self.nb_layers=nb_layers
        if self.bottleneck:
            self.nb_layers=self.nb_layers // 2
        self.nb_layer_spec=[self.nb_layers for _ in range(nb_dense)]
        # weights and dropouts
        self.dropout_rate=dropout_rate
        self.weight_decay=weight_decay
        if self.weight_decay:
            self.regularizer=l2(l=self.weight_decay)
        else:
            self.regularizer=None
    
    def __transitionBlock(self,X,b_idx):
        
        X = BatchNormalization(gamma_regularizer=self.regularizer,
                               beta_regularizer =self.regularizer,
                               name='TRANS_BLOCK_{}_BATCH_NORM'.format(b_idx))(X)
        
        X = Activation("relu",
                        name='TRANS_BLOCK_{}_RELU'.format(b_idx))(X)
        
        X = Conv2D(self.nb_filter, 
                    (1, 1), 
                    padding='same',
                    use_bias=False, 
                    kernel_regularizer=self.regularizer,
                    name='TRANS_BLOCK_{}_CONV2D'.format(b_idx))(X) 
        
        if self.dropout_rate:
            X = Dropout(self.dropout_rate,
                        name='TRANS_BLOCK_{}_DROPOUT'.format(b_idx))(X)
        
        X = AveragePooling2D((2, 2), 
                            strides=(2, 2),
                            name='TRANS_BLOCK_{}_AVGPOOL'.format(b_idx))(X)
        
        return X
        
    
    def __convBlock(self,X,b_idx,l_idx):
        if self.bottleneck:
            X=BatchNormalization(gamma_regularizer=self.regularizer,
                                beta_regularizer =self.regularizer,
                                name='BOTTLENECK_BATCH_NORM_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)
        
            X = Activation("relu",
                            name='BOTTLENECK_RELU_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)
       
            X = Conv2D(self.nb_filter*4, 
                       (1,1), 
                       padding='same',
                       use_bias=False, 
                       kernel_regularizer=self.regularizer,
                       name='BOTTLENECK_CONV2D_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)  
       
            if self.dropout_rate:
                X = Dropout(self.dropout_rate,
                            name='BOTTLENECK_DROPOUT_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)
            
        X=BatchNormalization(gamma_regularizer=self.regularizer,
                             beta_regularizer =self.regularizer,
                             name='BATCH_NORM_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)
        
        X = Activation("relu",
                        name='RELU_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)
       
        X = Conv2D(self.nb_filter, 
                    (3,3), 
                    padding='same',
                    use_bias=False, 
                    kernel_regularizer=self.regularizer,
                    name='CONV2D_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)  
       
        if self.dropout_rate:
            X = Dropout(self.dropout_rate,
                        name='DROPOUT_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(X)
       
        return X
    
    def __denseBlock(self,X,layer_count,b_idx):
        __feats=[X]
        
        for l_idx in range(1,layer_count+1):
            
            X_C=self.__convBlock(X,b_idx,l_idx)

            __feats.append(X_C)

            X = Concatenate(axis=-1,
                            name='CONCAT_BLOCK_{}_LAYER_{}'.format(b_idx,l_idx))(__feats)
       
            self.nb_filter+=self.growth_rate
        
        return X

    def get_model(self):
        #_initialConvLayer
        X=Conv2D(self.nb_filter*2,
                           (3,3), 
                            padding='same',
                            use_bias=False, 
                            kernel_regularizer=self.regularizer,
                            name='INITAIAL_CONV2D_LAYER')(self.IN)
                            

        # dense blocks
        for idx_dense in range(self.nb_dense - 1):
       
            X=self.__denseBlock(X,self.nb_layer_spec[idx_dense],idx_dense+1)
       
            X=self.__transitionBlock(X,idx_dense+1)

            self.nb_filter=int(self.nb_filter*self.compression)
        
        X=self.__denseBlock(X,self.nb_layer_spec[-1],self.nb_dense)
       
        X=BatchNormalization(gamma_regularizer=self.regularizer,
                             beta_regularizer =self.regularizer,
                             name='BATCH_NORM_FINAL')(X)
        
        X=Activation('relu',
                     name='FINAL_RELU')(X)
        
        X=GlobalAveragePooling2D(name='GLOBAL_POOL')(X)
        
        X=Dense(self.nb_classes, 
                activation='softmax',
                kernel_regularizer=self.regularizer,
                bias_regularizer =self.regularizer,
                name="CLASS_DENSE")(X)
        
        # naming
        
        self.model_name="DenseNet_blocks:{}_layers:{}_filters:{}_growth:{}_weightDecay:{}_dropout:{}_compression:{}_bottleneck:{}".format(self.nb_dense,
                                                                                                 self.nb_layers,
                                                                                                 self.STARTING_FILTER,
                                                                                                 self.growth_rate,
                                                                                                 self.weight_decay,
                                                                                                 self.dropout_rate,
                                                                                                 self.compression,
                                                                                                 self.bottleneck
                                                                                                )
        model=Model(inputs=self.IN,outputs=X,name='DenseNet')
        return model
#--------------------------------------------------------------------------------------
if __name__=='__main__':
    img_path='/home/ansary/RESEARCH/MEDICAL/pySKIND/info/'
    
    model=conv_net()
    model.summary()
    plot_model(model,to_file=os.path.join(img_path,'convNet.png'),show_layer_names=True,show_shapes=True)
    OBJ=DenseNet()
    model=OBJ.get_model()
    model.summary()
    plot_model(model,to_file=os.path.join(img_path,"{}.png".format(OBJ.model_name)),show_layer_names=True,show_shapes=True)
    