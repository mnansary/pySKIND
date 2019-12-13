#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

# import the library
from appJar import gui
import os
# create a GUI variable called app
app = gui()
#--------------------------------------------------------------------------------------------------------------------------------------------------
from PIL import Image, ImageTk
from coreLib.models import DenseNet
import numpy as np
#--------------------------------------------------------------------------------------------------------------------------------------------------
model_path=os.path.join(os.getcwd(),'models','denseNet_final.h5')
OBJ=DenseNet()
model=OBJ.get_model()
model.load_weights(model_path)
model.summary()

CLASSES=['eczema','psoriasis']
#--------------------------------------------------------------------------------------------------------------------------------------------------
def press(button):
    if button == "Exit":
        app.stop()
    else:
        img_path=app.getEntry("Image Location:")
        try:
            app.setImage("pic", img_path)
            app.setLabel("log", "LOG:Loaded Image")
            img=Image.open(img_path)
            img=img.resize((64,64))
            x=np.array(img)
            tensor=np.expand_dims(x,axis=0)
            tensor=tensor.astype('float32')/255.0
            pred=np.argmax(model.predict(tensor))
            app.setLabel("log", "LOG:DISEASE:{}".format(CLASSES[pred]))
        except:
            app.setLabel("log", "LOG:Path is Not Image File")
#--------------------------------------------------------------------------------------------------------------------------------------------------

back_path=os.path.join(os.getcwd(),'info','appback.png')
app.setFont(size=20, family="Times",weight="bold")
app.addImage("pic", back_path)
app.setImageSize("pic", 512, 512)
app.addLabelEntry("Image Location:")
app.addFlashLabel("log", "Program is running")
app.addButtons(["Predict", "Exit"], press)
if __name__=='__main__':
    app.go()