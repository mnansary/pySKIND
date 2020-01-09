# pyCONVCLASSIFIER

    Version: 0.0.5    
    Author : Md. Nazmuddoha Ansary, Shakir Hossain
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
* numpy==1.16.4  
* tensorflow==1.13.1        
* Keras==2.3.1        
* Python == 3.6.8
> Create a Virtualenv and *pip3 install -r requirements.txt*

#  Preprocessing
* Check The following Values in **config.json** 

        {
            "FLAGS":
            {
                "DATA_DIR"        : "/home/ansary/RESEARCH/MEDICAL/Data/",
                "OUTPUT_DIR"      : "/home/ansary/RESEARCH/MEDICAL/Data/",
                "CLASSES"         : "eczema,psoriasis",
                "IMAGE_DIM"       : 64,
                "ROT_ANGLES"      : "5,10,15,20,30,45,60,75",
                "EVAL_SPLIT"      : 0.05,
                "TEST_SPLIT"      : 0.2,
                "BATCH_SIZE"      : 128
            }
            
        } 


* **DATA_DIR** should have the following folder tree:
                
                Data
                ├── eczema
                └── psoriasis 

* run **main.py**

**Results**
* If execution is successful a folder called **OUTPUT_DIR/DataSet** should be created with the following folder tree:

        ├── Images
        │   ├── {Xid}_{xrot}_{xflip}_{xsample}_{xlabel}.png
        │   ├── {Xid}_{xrot}_{xflip}_{xsample}_{xlabel}.png
        ------------------------------------
        ------------------------------------
        │   ├── {Xid}_{xrot}_{xflip}_{xsample}_{xlabel}.png
        │   └── {Xid}_{xrot}_{xflip}_{xsample}_{xlabel}.png
        ├── Test
        │   ├── X_test.h5
        │   └── Y_test.h5
        └── Train
            ├── X_eval.h5
            ├── X_train.h5
            ├── Y_eval.h5
            └── Y_train.h5


*    Train Data Size:**24192** images
*    Test Data Size: **6272** images
*    Eval Data Size: **1152** images


**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  


# TPU(Tensor Processing Unit)
![](/info/src_img/tpu.ico?raw=true)*TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the **Tesla K80** available in Google Colab delivers respectable **1.87 TFlops** and has **12GB RAM**, the **TPUv2** available from within Google Colab comes with a whopping **180 TFlops**, give or take. It also comes with **64 GB** High Bandwidth Memory **(HBM)**.*
[Visit This For More Info](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)  


# Training

## MODEL: CONV_NET
* in **convNet.ipynb** : SECTION: *ConvNet Model Training* SUB-SECTION: **Data**, change the values of the following params as needed:

![](/info/colab_convNet.png?raw=true)

* [Model Structre](https://github.com/mnansary/pySKIND/blob/master/info/convNet.png)
* run **convNet.ipynb** in *colab*

![](/info/convNet_history.png?raw=true)

* F1 SCORE: **98.4375 %** 
## MODEL: DenseNet
The model is based on the original paper:[Densely Connected Convolutional Networks](https://ieeexplore.ieee.org/document/8099726)  
> Authors and Researchers: Gao Huang ; Zhuang Liu ; Laurens van der Maaten ; Kilian Q. Weinberger

The paper introduces **Dense Blocks** within the traditional convolutional neural network architechture.  
![](/info/dense1.png?raw=true)
The composite layers can also contain **bottoleneck layers**   
![](/info/dense2.png?raw=true)

As compared to well established CNN models (like : *FractNet* or *ResNet*) DenseNet has:  
    *   Less number of feature vector  
    *   Low information bottoleneck   
    *   Better Handling Of the *vanishing gradient* problem      


* [Default Model Structre](https://github.com/mnansary/pySKIND/blob/master/info/DenseNet.png)

* In addition to the previous parameters like **convNet**, the following valuese *may be changed* for *exploring*.

![](/info/colab_denseNet.png?raw=true)

* run **denseNet.ipynb** in *colab*

![](/info/denseNet_history.png?raw=true)

* F1 SCORE: **99.1390306122449 %** 

# Deployment

* run **app.py**
* used model: **DenseNet**
* After Training in **colab**,download the weights from **google drive** and place them in  **models** folder in the local copy of the repo

![](/info/ui.gif?raw=true)
