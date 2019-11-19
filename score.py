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
from sklearn import metrics
from coreLib.models import conv_net
from coreLib.utils import readh5,readJson,LOG_INFO
#-----------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='F1 Score of models')
parser.add_argument("model_name",help='name of the model to be scored. Available: convNet')
parser.add_argument("model_dir",help='model weights directory')
parser.add_argument("x_test",help='feature data path(h5)')
parser.add_argument("y_test",help='label data path(h5)')
args = parser.parse_args()
#-----------------------------------------------------Load Config----------------------------------------------------------
config_data = readJson('config.json')
IMAGE_DIM   = config_data['FLAGS']['IMAGE_DIM']
#-----------------------------------------------------------------------------------------------------------------------------------

def score_model(args):
    LOG_INFO('Loading Model')
    model_name=args.model_name
    if model_name=='convNet':
        model=conv_net(img_dim=IMAGE_DIM)
        model.load_weights(args.model_dir)
    else:
        raise ValueError('Check model_name.')
    x_test=readh5(args.x_test)
    x_test=x_test.astype('float32')/255.0
    y_test=readh5(args.y_test)
    y_true =[np.argmax(y) for y in y_test]
    
    LOG_INFO('Getting Predictions')
    _pbar=ProgressBar()
    y_pred =[np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in _pbar(x_test)]
    f1_accuracy =100* metrics.f1_score(y_true,y_pred, average = 'micro')
    LOG_INFO('F1 SCORE:{} % '.format(f1_accuracy))

if __name__ == "__main__":
    score_model(args)