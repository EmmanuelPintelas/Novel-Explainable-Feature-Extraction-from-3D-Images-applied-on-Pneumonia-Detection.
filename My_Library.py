#___________________________________________
# Creator Emmanuel Pintelas
#___________________________________________
# https://www.kaggle.com/c/deepfake-detection-challenge/discussion/128954
from sklearn.model_selection import KFold
import gc
import os
from random import shuffle
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Lambda,Activation,Input,Flatten,Reshape,Conv2DTranspose
import skimage
from skimage.exposure import rescale_intensity
from tensorflow import keras
from tensorflow.keras import layers
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, GlobalMaxPooling2D
import psutil
import multiprocessing as mp
from tensorflow.keras.applications import ResNet50, DenseNet201, Xception, VGG16, InceptionV3, InceptionResNetV2, MobileNetV2, NASNetMobile
import copy
mp.cpu_count()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from  skimage.feature import greycoprops
#%matplotlib inline
#import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Lambda,Activation,Input,Flatten,Reshape,Conv2DTranspose, LeakyReLU
import tensorflow.keras.backend as K
#from keras.layers.merge import add
from sklearn.model_selection import train_test_split
import os
import glob
from time import time,asctime
from random import randint as r
import random
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from numpy import save, load
from   sklearn.metrics           import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from numpy import load
import pandas as pd
import skimage
from skimage.transform import resize
import math
from sklearn.svm import SVC
from tensorflow.keras.layers import Input, Dense, UpSampling2D, Flatten, Reshape, LSTM
from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage import img_as_float32
from concurrent.futures import ThreadPoolExecutor
import scipy.stats
import nibabel as nib

if 1==1:
    import ID3                         
    from ID3 import *







#_______________________________________________________________________________________________________________________________________________________________
#_____________ MEDICAL 3D CNN CLASSIFICATION FUNCTIONS _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________


if 1==2:

    # CT_0_path = [
    #     os.path.join(os.getcwd(), "APOLLO-5-LSCC/AP-6H6G", x)
    #     for x in os.listdir("APOLLO-5-LSCC/AP-6H6G")
    # ]



    CT_0_path = [
        os.path.join(os.getcwd(), "COVID19_1110/studies/CT-0", x)
        for x in os.listdir("COVID19_1110/studies/CT-0")
    ] #[:2]

    CT_1_path = [
        os.path.join(os.getcwd(), "COVID19_1110/studies/CT-1", x)
        for x in os.listdir("COVID19_1110/studies/CT-1")
    ]#[:2]
    CT_2_path = [
        os.path.join(os.getcwd(), "COVID19_1110/studies/CT-2", x)
        for x in os.listdir("COVID19_1110/studies/CT-2")
    ]#[:2]
    CT_3_path = [
        os.path.join(os.getcwd(), "COVID19_1110/studies/CT-3", x)
        for x in os.listdir("COVID19_1110/studies/CT-3")
    ]#[:2]
    CT_4_path = [
        os.path.join(os.getcwd(), "COVID19_1110/studies/CT-4", x)
        for x in os.listdir("COVID19_1110/studies/CT-4")
    ]#[:2]
    CT_1_path, CT_2_path, CT_3_path, CT_4_path = pd.DataFrame(CT_1_path), pd.DataFrame(CT_2_path), pd.DataFrame(CT_3_path), pd.DataFrame(CT_4_path)

    CT_N_path = pd.DataFrame(CT_0_path).values
    CT_VP_path234 = np.concatenate((CT_2_path, CT_3_path, CT_4_path),axis=0)


    CT_1_path = CT_1_path.values
    he,wid,dep = 300,300,80
    #Scans = []
    _cnt__ = 0
    for k in range(len(CT_1_path)):

        k = k + 271

        scan = nib.load(CT_1_path[k][0])
        scan = scan.get_fdata()

        # mn = np.min(scan)
        # scan = scan-mn
        # mx = np.max(scan)
        # scan = scan/mx
        scan = resize(scan,(he,wid,dep))
        scan = scan.reshape(he,wid,dep,1)
        scan = np.concatenate((scan,scan,scan),axis=3)

        # fig=plt.figure(figsize=(12,12))
        # rows, columns = 10,8
        # for i in range(1,  dep+1):  
        #             img = scan[i-1,:,:] 
        #             fig.add_subplot(rows, columns, i)
        #             plt.axis('off')
        #             plt.imshow(img)
        # plt.show()

        #Scans.append(scan)

        _cnt__+=1
        os.mkdir('CT_1/'+str(k+1))

        cnt = 0
        for __i__ in range (dep) :
            fr = scan[:,:,__i__,:]
            cnt+=1
            #fr = np.uint8(fr)
            skimage.io.imsave ('CT_1/'+str(k+1)+'/'+str(cnt)+'.jpg',fr)
            ##img = io.imread('CT_1/'+str(_cnt__)+'/'+str(cnt)+'.jpg')

            # plt.figure()
            # plt.imshow(img)
            # plt.show()


        # fig=plt.figure(figsize=(12,12))
        # rows, columns = 10,8
        # for i in range(1,  dep+1):  
        #             img = scan[i-1,:,:] 
        #             fig.add_subplot(rows, columns, i)
        #             plt.axis('off')
        #             plt.imshow(img)
        # plt.show()

    s = 1

def data_3d_loading(vid_path, divider, heigth, width):

                    vid_path = vid_path[0][0]
                    Frames_id = os.listdir(vid_path)
                    video = []
                    for fr_int in range(len(Frames_id)):
                                    fr_str = str(fr_int+1)+'.jpg'
                                    img_path = vid_path+'/'+fr_str
                                    img = io.imread(img_path) # all frames must be of same size
                                    img = img.astype(float)
                                    img = img/255.
                                    #img = img.astype(float)

                                    img = cv2.resize(img, (heigth, width))
                                    video.append(img)  
                                    # plt.figure()
                                    # plt.imshow(img)
                                    # plt.show()

                    video = np.array(video)

                    video_sampled = []
                    init_depth = video.shape[0]
                        #divider = int (init_depth/depth)
                    for _i in range(init_depth):
                                if _i % divider == 0:
                                    video_sampled.append(video[_i])

                                    # plt.figure()
                                    # plt.imshow(video[_i])
                                    # plt.show()

                    video_sampled = np.array(video_sampled)

                    return video_sampled

class Data_Generator_3D_CNN(tf.keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self,mode='',
                    batch_size=1, divider='', width='', heigth='', paths = '', labels = '', n_channels=3, shuffle=False):

            'Initialization'
            self.mode = mode
            self.batch_size = batch_size
            self.paths = paths   # video paths of jpg frame images
            self.labels = labels # video labels
            self.divider = divider   # Number of frames
            self.width = width
            self.heigth = heigth
            self.n_channels = n_channels
            #self.shuffle = shuffle
            #self.augment = augmentations
            self.on_epoch_end()
            self.cnt = 0


        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.ceil(len(self.paths) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.paths))]
            # Find list of IDs
            list_IDs_im = [self.paths[k] for k in indexes]

            Y = np.empty((len(list_IDs_im),2))
            j=-1
            for i in indexes:
                j+=1
                Y[j,] = self.labels[i] 
                #print(i)

            # Generate data
            Χ   = self.data_generation(list_IDs_im)
            
            self.cnt +=1
            #print (self.cnt)

            if self.mode == 'predict':
                return Χ   
            elif self.mode == 'train_C': 
                return Χ, Y      
            elif self.mode == 'train_AE': 
                Χ1 = Χ
                Χ2 = np.copy(Χ1)
                return Χ1, Χ2  

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.paths))
            # if self.shuffle == True:
            #     np.random.shuffle(self.indexes)

        def data_generation(self, list_IDs_im):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization

            depth = 80/self.divider
            X = np.empty((len(list_IDs_im),int(depth),self.width,self.heigth,self.n_channels))

            # Generate data
            for i, vid_path in enumerate(list_IDs_im):  
                vid_path = vid_path[0]
                X[i,] = data_3d_loading(vid_path, self.divider, self.heigth, self.width)

                # plt.figure()
                # plt.imshow(X[0][0])
                # plt.show()

            return X





if 1==2:

    CT_VP_tr_path = [
        os.path.join(os.getcwd(), "Dataset_Medical/Train/CT_VP_path234", x)
        for x in os.listdir("Dataset_Medical/Train/CT_VP_path234")
    ] 
    CT_N_tr_path = [
        os.path.join(os.getcwd(), "Dataset_Medical/Train/CT_N_path", x)
        for x in os.listdir("Dataset_Medical/Train/CT_N_path")
    ] 

    CT_VP_ts_path = [
        os.path.join(os.getcwd(), "Dataset_Medical/Test/CT_VP_path234", x)
        for x in os.listdir("Dataset_Medical/Test/CT_VP_path234")
    ] 
    CT_N_ts_path = [
        os.path.join(os.getcwd(), "Dataset_Medical/Test/CT_N_path", x)
        for x in os.listdir("Dataset_Medical/Test/CT_N_path")
    ] 

    VP_tr_labels = np.array([1 for _ in range(len(CT_VP_tr_path))])
    N_tr_labels = np.array([0 for _ in range(len(CT_N_tr_path))])
    VP_ts_labels = np.array([1 for _ in range(len(CT_VP_ts_path))])
    N_ts_labels = np.array([0 for _ in range(len(CT_N_ts_path))])


    train_paths = np.concatenate((pd.DataFrame(CT_VP_tr_path), pd.DataFrame(CT_N_tr_path)), axis=0)
    test_paths = np.concatenate((pd.DataFrame(CT_VP_ts_path), pd.DataFrame(CT_N_ts_path)), axis=0)
    Labels_train = np.concatenate((VP_tr_labels, N_tr_labels), axis=0)
    Labels_test = np.concatenate((VP_ts_labels, N_ts_labels), axis=0)


    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder()
    Labels_train_oh = onehot_encoder.fit_transform(Labels_train.reshape(-1,1)).toarray()#.astype(int)
    Labels_test_oh = onehot_encoder.fit_transform(Labels_test.reshape(-1,1)).toarray()#.astype(int)


    from random import shuffle
    ind_list = [_r_ for _r_ in range(len(train_paths))]
    shuffle(ind_list)
    train_paths  = train_paths[ind_list]
    Labels_train_oh  = Labels_train_oh[ind_list]
    Labels_train  = Labels_train[ind_list]









#_______________________________________________________________________________________________________________________________________________________________
#_____________ VIDEO 3D CNN CLASSIFICATION FUNCTIONS _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________

class Video_Generator_3D_CNN(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,mode='',
                 batch_size=1, depth='', width='', heigth='', paths = '', labels = '', n_channels=3, shuffle=False):

        'Initialization'
        self.mode = mode
        self.batch_size = batch_size
        self.paths = paths   # video paths of jpg frame images
        self.labels = labels # video labels
        self.depth = depth   # Number of frames
        self.width = width
        self.heigth = heigth
        self.n_channels = n_channels
        #self.shuffle = shuffle
        #self.augment = augmentations
        self.on_epoch_end()
        self.cnt = 0


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.paths))]
        # Find list of IDs
        list_IDs_im = [self.paths[k] for k in indexes]

        Y = np.empty((len(list_IDs_im),2))
        j=-1
        for i in indexes:
            j+=1
            Y[j,] = self.labels[i] 
            #print(i)

        # Generate data
        Χ   = self.data_generation(list_IDs_im)
        
        

        

        self.cnt +=1
        #print (self.cnt)

        if self.mode == 'predict':
               return Χ   
        elif self.mode == 'train_C': 
            return Χ, Y      
        elif self.mode == 'train_AE': 
            Χ1 = Χ
            Χ2 = np.copy(Χ1)
            return Χ1, Χ2  

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im),self.depth,self.width,self.heigth,self.n_channels))

        # Generate data
        for i, vid_path in enumerate(list_IDs_im):  
            vid_path = vid_path[0]
            Frames_id = os.listdir(vid_path)

            Frames_id_int = []
            for fr in Frames_id:
                    if '_' not in fr:    
                                fr = int (fr.replace('.jpg',''))
                                Frames_id_int.append (fr)
            Frames_id_int = np.array(Frames_id_int)
            Frames_id_int = np.sort(Frames_id_int) 

            signal = 0
            while signal != 1:
                    try:
                        for fr_int in Frames_id_int:
                            fr_str = str(fr_int)+'.jpg'
                            img_path = vid_path+'/'+fr_str
                            img = io.imread(img_path) # all frames must be of same size
                            img = img.astype(float)
                            Frames_per_Instance.append(img)  
                        Frames_per_Instance = np.array(Frames_per_Instance)
                        signal = 1
                    except:  
                        Frames_per_Instance = []


            #Frames_per_Instance = resize (Frames_per_Instance,(self.depth,self.width,self.heigth,self.n_channels))
            Frames_per_Instance_Sampled = np.zeros((self.depth,self.width,self.heigth,self.n_channels))
            init_depth = Frames_per_Instance.shape[0]
            sampling_Ratio = int (init_depth/self.depth)
            j=-1
            for _i in range(300):
                    if _i % sampling_Ratio == 0:
                        j+=1
                        Frames_per_Instance_Sampled[j] = cv2.resize(Frames_per_Instance[_i],(self.width,self.heigth)).astype(float)/255.
                        # plt.figure()
                        # plt.imshow(Frames_per_Instance_Sampled[j])
                        # plt.show()

            X[i,] = Frames_per_Instance_Sampled

        

        return X





def Build_Classifier_3D_CNN (depth, width, heigth):

                    inputs = keras.Input((depth, width, heigth, 3))

                    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
                    x = layers.MaxPool3D(pool_size=(1,2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
                    x = layers.MaxPool3D(pool_size=(1,2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
                    x = layers.MaxPool3D(pool_size=(1,2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
                    x = layers.MaxPool3D(pool_size=(1,2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.GlobalAveragePooling3D()(x)
                    # FC Layer ___________
                    x = layers.Dense(units=64, activation="relu")(x)
                    x = layers.Dropout(0.2)(x)

                    outputs = layers.Dense(units=2, activation="sigmoid")(x)

                    # Define the model.
                    model3d = keras.Model(inputs, outputs, name="Classifier_3D_CNN")

                    return model3d

def Build_MI_CNN_2D (width, heigth, channels):
                channels = 3
                inputs1 = keras.Input((width, heigth, channels))
                inputs2 = keras.Input((width, heigth, channels))
                inputs3 = keras.Input((width, heigth, channels))
                def Block (inputs):
                    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
                    x = layers.MaxPool2D(pool_size=(2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
                    x = layers.MaxPool2D(pool_size=(2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
                    x = layers.MaxPool2D(pool_size=(2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
                    x = layers.MaxPool2D(pool_size=(2,2))(x)
                    x = layers.BatchNormalization()(x)

                    x = layers.GlobalAveragePooling2D()(x)
                    return x

                x1 = Block (inputs1)
                x2 = Block (inputs2)
                x3 = Block (inputs3)
                x_xon = tf.concat(axis=1,values=[x1, x2, x3])

                x = layers.Dense(units=64, activation="relu")(x_xon)
                x = layers.Dropout(0.2)(x)
                outputs = layers.Dense(units=2, activation="sigmoid")(x)

                model_MI = keras.Model([inputs1,inputs2,inputs3], outputs, name="Classifier_2D_MI_CNN")

                return model_MI
#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________








#_______________________________________________________________________________________________________________________________________________________________
#_____________ CNN Feature Extractor _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________


def Build_CNN_F_Extr (CNN,layer_id):
        layer =  CNN.layers[layer_id]
        layer_name = layer.name
        CNN_F_Extr = Model(inputs=CNN.input, outputs=CNN.get_layer(layer_name).output)
        model = Sequential()
        model.add(CNN_F_Extr)
        model.add(Flatten())
        CNN_F_Extr = model
        return CNN_F_Extr

#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________















#_______________________________________________________________________________________________________________________________________________________________
#_____________ VIDEO 2D CNN FUNCTIONS _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________
 
def Build_R2D_Classifier (width, heigth):

    R2D = ResNet50(include_top=False, weights='imagenet',  input_shape=(width, heigth,3))
    model = Sequential()
    model.add(R2D)
    model.add(GlobalAveragePooling2D())
    # FC Layer ___________
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(layers.Dense(units=2, activation="sigmoid"))
    R2D_Classifier = model
    R2D_Classifier.summary()
    return R2D_Classifier






def Build_I2D_Classifier (width, heigth):

    I2D = InceptionV3(include_top=False, weights='imagenet',  input_shape=(width, heigth,3))
    model = Sequential()
    model.add(I2D)
    model.add(GlobalAveragePooling2D())
    # FC Layer ___________
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(layers.Dense(units=1, activation="sigmoid"))
    I2D_Classifier = model
    I2D_Classifier.summary()
    return I2D_Classifier


def Video_2D_CNN_FE (CNN2D_FE,divider,width,heigth,n_channels,paths):
    #FEATURES = np.zeros(((paths.shape[0]), depth, CNN2D_FE.output.shape[1]))
    depth = int(300/divider)
    FEATURES = np.zeros(((paths.shape[0]),  depth, CNN2D_FE.output_shape[1]))
    index = -1
    for path in paths:
        index+=1
        path = path.reshape(-1,1)

        # Compression Time ____
        video = video_loading (path,heigth, width) #resize
        video = Video_Subasampling (video, divider)#subsamble
        # Compression Time ____

        # FE Time ____
        video_f = CNN2D_FE.predict(video)
        # FE Time ____
                    
        FEATURES[index] = video_f  
    return FEATURES


#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________












#_______________________________________________________________________________________________________________________________________________________________
#_____________ CAE 3D  _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________


            
def Build_AE3D_EE(depth, width, heigth):
                    inputs = keras.Input((int(depth), width, heigth, 3)) # init 300,150,150

                    x = layers.Conv3D(filters=16, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(inputs)
                    x = layers.LeakyReLU()(x)
                    x = layers.MaxPool3D(pool_size=(3,1,1))(x)
                    
                    x = layers.Conv3D(filters=3, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(x)
                    x = layers.LeakyReLU()(x)
            
    
    
                    x = tf.keras.layers.Conv3DTranspose (filters=32, kernel_size=(3,3,3), strides=(3,1,1), padding='same')(x) 
                    x = layers.LeakyReLU()(x)
                    x = tf.keras.layers.Conv3DTranspose (filters=16, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(x) 
                    x = layers.LeakyReLU()(x)
                    
                    outputs =  layers.Conv3DTranspose(filters=3, kernel_size=(3,3,3), strides=(1,1,1), activation="sigmoid", padding='same')(x)

                    # Define the model.
                    ae3d = keras.Model(inputs, outputs, name="ae_3D")
                    
                    return ae3d

# CAE = Build_AE3D_EE(depth, width, heigth)
# CAE.summary()


#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________










#_______________________________________________________________________________________________________________________________________________________________
#_____________ HC FEATURES 1 _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________




def Co_Oc_Features (img):
    img = img*255
    img = np.uint8(img)
    if img.shape[-1] == 3:
                    matrix0 = skimage.feature.texture.greycomatrix(img[:,:,0], [1,2], [0, np.pi/2],  normed=True, symmetric=True)
                    matrix1 = skimage.feature.texture.greycomatrix(img[:,:,1], [1,2], [0, np.pi/2],  normed=True, symmetric=True)
                    matrix2 = skimage.feature.texture.greycomatrix(img[:,:,2], [1,2], [0, np.pi/2],  normed=True, symmetric=True)
                    con0,dis0,hom0,ASM0,en0,cor0 = greycoprops(matrix0, 'contrast'), greycoprops(matrix0, 'dissimilarity'), greycoprops(matrix0, 'homogeneity'),greycoprops(matrix0, 'ASM'),greycoprops(matrix0, 'energy'),greycoprops(matrix0, 'correlation')
                    con1,dis1,hom1,ASM1,en1,cor1 = greycoprops(matrix1, 'contrast'), greycoprops(matrix1, 'dissimilarity'), greycoprops(matrix1, 'homogeneity'),greycoprops(matrix1, 'ASM'),greycoprops(matrix1, 'energy'),greycoprops(matrix1, 'correlation')
                    con2,dis2,hom2,ASM2,en2,cor2 = greycoprops(matrix2, 'contrast'), greycoprops(matrix2, 'dissimilarity'), greycoprops(matrix2, 'homogeneity'),greycoprops(matrix2, 'ASM'),greycoprops(matrix2, 'energy'),greycoprops(matrix2, 'correlation')
                    con0,dis0,hom0,ASM0,en0,cor0, con1,dis1,hom1,ASM1,en1,cor1, con2,dis2,hom2,ASM2,en2,cor2 = con0.reshape(1,4),dis0.reshape(1,4),hom0.reshape(1,4),ASM0.reshape(1,4),en0.reshape(1,4),cor0.reshape(1,4), con1.reshape(1,4),dis1.reshape(1,4),hom1.reshape(1,4),ASM1.reshape(1,4),en1.reshape(1,4),cor1.reshape(1,4), con2.reshape(1,4),dis2.reshape(1,4),hom2.reshape(1,4),ASM2.reshape(1,4),en2.reshape(1,4),cor2.reshape(1,4)
                    
                    mean0,std0, mean1,std1, mean2,std2 = np.array(np.mean(img[:,:,0])).reshape(-1,1),np.array(np.std(img[:,:,0])).reshape(-1,1), np.array(np.mean(img[:,:,1])).reshape(-1,1),np.array(np.std(img[:,:,1])).reshape(-1,1), np.array(np.mean(img[:,:,2])).reshape(-1,1),np.array(np.std(img[:,:,2])).reshape(-1,1)
                    
                    co_oc = np.concatenate((mean0,std0, mean1,std1, mean2,std2, con0,dis0,hom0,ASM0,en0,cor0, con1,dis1,hom1,ASM1,en1,cor1, con2,dis2,hom2,ASM2,en2,cor2),axis = 1) 
                    return co_oc
    else:
        matrix0 = skimage.feature.texture.greycomatrix(img, [1,2], [0, np.pi/2],  levels=256, normed=True, symmetric=True)
        con0,dis0,hom0,ASM0,en0,cor0 = greycoprops(matrix0, 'contrast'), greycoprops(matrix0, 'dissimilarity'), greycoprops(matrix0, 'homogeneity'),greycoprops(matrix0, 'ASM'),greycoprops(matrix0, 'energy'),greycoprops(matrix0, 'correlation')
        co_oc = np.concatenate((con0.reshape(1,4),dis0.reshape(1,4),hom0.reshape(1,4),ASM0.reshape(1,4),en0.reshape(1,4),cor0.reshape(1,4)),axis = 1) 
        return co_oc



def Number_Contours (contours, hierarchy_contours):
    
    if len(contours)==0:
        number_contours_OTSU = 0
    else:
        number_contours_OTSU = hierarchy_contours.shape[1] 
    return number_contours_OTSU

def _Area_ (contours, hierarchy_contours):

    if len(contours)==0:
        number_contours_OTSU = 0
    else:
        number_contours_OTSU = hierarchy_contours.shape[1] 

    Features = []
    number_contours = number_contours_OTSU
    ccontours = contours

    if number_contours > 1:

            ss = ccontours[0][:]
            if 223 and 0 in ss:
                number_contours = number_contours - 1
                discrete_contours = ccontours[1:]
            else:
                discrete_contours = ccontours


            number_of_very_small_contours = 0
            number_of_small_contours = 0
            number_of_medium_contours = 0
            number_of_big_contours = 0
            number_of_very_big_contours = 0
            ind = -1
            Perimeters = np.zeros((number_contours,1))
            for _contour in discrete_contours:
                ind += 1
                Perimeters[ind] = cv2.arcLength(_contour,True)



            max_perimeter = np.max(Perimeters)
            mean_perimeter = np.mean(Perimeters)
            std_perimeter = np.std(Perimeters)


            for perimeter in Perimeters:
                _ = perimeter
                if _ <= 0.2*(max_perimeter) and _ >= 0.0*(max_perimeter):
                    number_of_very_small_contours += 1
                if _ <= 0.4*(max_perimeter) and _ >= 0.2*(max_perimeter):
                    number_of_small_contours += 1
                if _ <= 0.6*(max_perimeter) and _ >= 0.4*(max_perimeter):
                    number_of_medium_contours += 1
                if _ <= 0.8*(max_perimeter) and _ >= 0.6*(max_perimeter):
                    number_of_big_contours += 1
                if _ <= 1.0*(max_perimeter) and _ >= 0.8*(max_perimeter):
                    number_of_very_big_contours += 1

            Features.append(math.log( number_of_very_small_contours+1))
            Features.append(math.log( number_of_small_contours+1))
            Features.append(math.log( number_of_medium_contours+1))
            Features.append(math.log( number_of_big_contours+1))
            Features.append(math.log( number_of_very_big_contours+1))

            Features.append(math.log( max_perimeter+1))
            Features.append(math.log( mean_perimeter+1))
            Features.append(math.log( std_perimeter+1))
    else:
            number_of_very_small_contours = 0
            number_of_small_contours = 0
            number_of_medium_contours = 0
            number_of_big_contours = 0
            number_of_very_big_contours = 0

            max_perimeter = 0
            mean_perimeter = 0
            std_perimeter = 0



            Features.append(math.log( number_of_very_small_contours+1))
            Features.append(math.log( number_of_small_contours+1))
            Features.append(math.log( number_of_medium_contours+1))
            Features.append(math.log( number_of_big_contours+1))
            Features.append(math.log( number_of_very_big_contours+1))

            Features.append(math.log( max_perimeter+1))
            Features.append(math.log( mean_perimeter+1))
            Features.append(math.log( std_perimeter+1))

    return Features


##BALE KAI TO m1 = embado


def Shapes (contours, hierarchy_contours):

    if len(contours)==0:
        number_contours_OTSU = 0
    else:
        number_contours_OTSU = hierarchy_contours.shape[1] 

    Features = []
    number_contours = number_contours_OTSU
    ccontours = contours

    if number_contours > 1:

            ss = ccontours[0][:]
            if 223 and 0 in ss:
                number_contours = number_contours - 1
                discrete_contours = ccontours[1:]
            else:
                discrete_contours = ccontours

            ind = -1
            M = np.zeros((number_contours,1))
            M2 = np.zeros((number_contours,31))
            N_VERTICES , MEANS, STDS, RATIO_BOX, RATIO_CYRCLE, RATIO_ELLIPSE, MASSES  = np.copy(M),np.copy(M),np.copy(M),np.copy(M),np.copy(M),np.copy(M),np.copy(M2)
            for _contour in discrete_contours:
                AREA = _contour.shape[0] + 1
                ind += 1

                if 1==1:
                    M = cv2.moments(_contour)           #
                    H =  list(cv2.HuMoments(M) [:,0]) 
                    M_ = []
                    for m in M:
                        M_.append(M[m])
                    jj = -1
                    for f in M_:
                        jj+=1
                        if f>0: 
                            MASSES[ind,jj] = math.log(f)
                        elif f == 0: 
                            MASSES[ind,jj] = 0
                        else: 
                            MASSES[ind,jj] = -math.log(-f)
    
                    for f in H:
                        jj+=1
                        if f>0: 
                            MASSES[ind,jj] = math.log(f)
                        elif f == 0: 
                            MASSES[ind,jj] = 0
                        else: 
                            MASSES[ind,jj] = -math.log(-f)







                _mean = np.mean (_contour)
                _std  = np.std (_contour)

                n_vertices = _contour.shape[0]
                N_VERTICES[ind] = n_vertices

                MEANS[ind] = _mean
                STDS[ind]  = _std


                # # # Does the shape looks like X known Shape? (value close to 1 means almost equal) 
                # Rectangle = cv2.minAreaRect(_contour)              # Does the shape looks like Rectangle?
                # box = cv2.boxPoints(Rectangle)
                # box = np.int0(box)
                # box_area = cv2.contourArea(box)
                # RATIO_BOX[ind] = AREA/(box_area+1)

                center, radius = cv2.minEnclosingCircle(_contour)  # Does the shape looks like Circle?
                cyrcle_area = 3.14*radius**2
                RATIO_CYRCLE[ind] = AREA/(cyrcle_area+1)

                # if n_vertices <5:
                #     RATIO_ELLIPSE[ind] = 0
                # else:
                #     Ellipse = cv2.fitEllipse(_contour)                # Does the shape looks like Ellipse?
                #     boxEllipse = cv2.boxPoints(Ellipse)
                #     boxEllipse = np.int0(boxEllipse)
                #     boxEllipse_area = cv2.contourArea(boxEllipse)
                #     RATIO_ELLIPSE[ind] = AREA/(boxEllipse_area+1)



            max_n_edges, mean_n_edges, std_n_edges = np.max(N_VERTICES), np.mean(N_VERTICES), np.std(N_VERTICES)

            masses_max, masses_min, masses_mean, masses_std = np.min(MASSES,axis=0), np.max(MASSES,axis=0), np.mean(MASSES,axis=0), np.std(MASSES,axis=0)
            for _ in masses_max:
                Features.append(_)
            for _ in masses_min:
                Features.append(_)
            for _ in masses_mean:
                Features.append(_)
            for _ in masses_std:
                Features.append(_)

            max_mean, mean_mean, std_mean = np.max(MEANS), np.mean(MEANS), np.std(MEANS) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # mean_mean = representative position of objects: center of gravity, 

            max_std,mean_std,std_std      = np.max(STDS), np.mean(STDS), np.std(STDS)
            #max_rectangle, mean_rectangle, std_rectangle = np.max(RATIO_BOX), np.mean(RATIO_BOX), np.std(RATIO_BOX)
            max_cyrcle,mean_cyrcle,std_cyrcle     = np.max(RATIO_CYRCLE), np.mean(RATIO_CYRCLE), np.std(RATIO_CYRCLE)
            #max_ellipse,mean_ellipse,std_ellipse      = np.max(RATIO_ELLIPSE), np.mean(RATIO_ELLIPSE), np.std(RATIO_ELLIPSE)

            for f in [max_n_edges, mean_n_edges, std_n_edges, max_mean, mean_mean, std_mean, max_std,mean_std,std_std,
                        max_cyrcle,mean_cyrcle,std_cyrcle]:
                Features.append(f)

    # else:
    #         for f in [0, 0, 0,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0]:
    #             Features.append(f)
    else:
            for i__ in range(136):
                Features.append(0)

    return Features





def Lines(im_Gray):
    #<------------- Number Lines   
    # In mathematics, a curve (also called a curved line in older texts) is an object similar to a line, 
    # but that does not have to be straight. 
    #---------------------------------------------------------------------------------------------------------------------
    # Trixes detector
    Features = []
    for dd in [0, 5, 15,25]:
        if dd>0: im_Gray = cv2.medianBlur(im_Gray, dd)
        canny_output = cv2.Canny(im_Gray, 100, 255)
        #plt.figure()
        #plt.imshow(canny_output)
        #plt.show()
        lines, hierarchy_lines  = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(lines) == 0:
            number_lines = 0
        else:
            number_lines = hierarchy_lines.shape[1]

        Features.append(math.log( number_lines+1))
    return Features


def Features_N_M_S_A (im_Binary):

    
    # Find Contours
    contours, hierarchy_contours  = cv2.findContours(im_Binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find Number of Contours
    if 1==1:
        Mean, Std = math.log( np.mean(im_Binary)+1), math.log( np.std(im_Binary)+1)
        Number_Contours_ = math.log( Number_Contours (contours, hierarchy_contours)+1)
        # Find Sizes of Contours
        Area_List =  _Area_(contours, hierarchy_contours)

        _f_features = []
        for f in Area_List:
            _f_features.append(f)
        _f_features.append(Number_Contours_)
        _f_features.append(Mean)
        _f_features.append(Std)

        # 11 (0...10)
        #[number_of_very_small_contours, 
        # number_of_small_contours, 
        # number_of_medium_contours, 
        # number_of_big_contours, 
        # number_of_very_big_contours, 
        # max_perimeter, 
        # mean_perimeter,
        # std_perimeter,
        # Number_Contours_,
        # Mean,
        # Std

    if 1==1:
        Shapes_List = Shapes(contours, hierarchy_contours)
        for f in Shapes_List:
            _f_features.append(f)
        # 12 (11...22)

        # 23 x 10 = 230
        #[max_n_edges, mean_n_edges, std_n_edges, max_mean, mean_mean, std_mean, max_std,mean_std,std_std, max_cyrcle,mean_cyrcle,std_cyrcle]
    return _f_features


def Other_Features (IMG) :

                #plt.figure()
                #plt.imshow(IMG)
                #plt.show()
                A = 0
                if IMG.shape[-1] == 3:
                    IMG = IMG[:,:,0]#IMG = IMG[:,:,0]
                #IMG = cv2.resize(IMG, (600,600))
                FINAL_FEATURES = []  
                if    np.max(IMG) <= 1.5: 
                        im_Gray = IMG * 255 
                else:
                    im_Gray = IMG
                if 1 == 2:
                    plt.figure()
                    plt.imshow(im_Gray)
                    plt.show()
                im_Gray = im_Gray.astype(np.uint8)
                #im_Gray = cv2.medianBlur(im_Gray,5)
                if A == 1:
                    plt.figure()
                    plt.imshow(im_Gray)
                    plt.show()
                vds = im_Gray
                mean_Gray = np.mean(vds[:,:])
                std_Gray = np.std(vds[:,:])

                
                #<------------- Number Contours
                #---------------------------------------------------------------------------------------------------------------------
                if 1==1:
                        im_Binary_GAUSSIAN = cv2.adaptiveThreshold(im_Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)  # 1
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary_GAUSSIAN)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary_GAUSSIAN)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)

                        im_Binary_GAUSSIAN = cv2.adaptiveThreshold(im_Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)  # 2
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary_GAUSSIAN)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary_GAUSSIAN)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)


                        im_Binary_GAUSSIAN = cv2.adaptiveThreshold(im_Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)  # 3
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary_GAUSSIAN)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary_GAUSSIAN)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)


                        im_Binary_GAUSSIAN = cv2.adaptiveThreshold(im_Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1) # 4
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary_GAUSSIAN)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary_GAUSSIAN)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)



                        im_Binary_GAUSSIAN = cv2.adaptiveThreshold(im_Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6) # 5
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary_GAUSSIAN)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary_GAUSSIAN)  # <<<<<<<<<<<<<<<<<<<<
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)


                        _, im_Binary_Stable_Th = cv2.threshold(im_Gray, 127, 255, cv2.THRESH_BINARY)    
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary_Stable_Th)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary_Stable_Th)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)

                        _, im_Binary1 = cv2.threshold(im_Gray, mean_Gray, 255, cv2.THRESH_BINARY)  # 7
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary1)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary1)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)

                        _, im_Binary2 = cv2.threshold(im_Gray, mean_Gray+std_Gray, 255, cv2.THRESH_BINARY) # 8
                        if A == 1:
                            plt.figure()
                            plt.imshow(im_Binary2)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary2)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)

                        if mean_Gray-std_Gray <= 0:
                            _, im_Binary3 = cv2.threshold(im_Gray, mean_Gray+0.5*std_Gray, 255, cv2.THRESH_BINARY)
                        else:
                            _, im_Binary3 = cv2.threshold(im_Gray, mean_Gray-std_Gray, 255, cv2.THRESH_BINARY) # 9
                        if 1 == 2:
                            plt.figure()
                            plt.imshow(im_Binary3)
                            plt.show()
                        _N_M_S_A_features = Features_N_M_S_A (im_Binary3)
                        for f in _N_M_S_A_features:
                            FINAL_FEATURES.append(f)


                _,im_Binary_OTSU = cv2.threshold(im_Gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # # 10
                if 1 == 2:
                    plt.figure()
                    plt.imshow(im_Binary_OTSU)
                    plt.show()
                _N_M_S_A_features = Features_N_M_S_A (im_Binary_OTSU)  #<<<<<<<<<<<<<<<<<<
                for f in _N_M_S_A_features:
                    FINAL_FEATURES.append(f)


                blur = cv2.GaussianBlur(im_Gray,(5,5),0)
                _,im_Binary_OTSU = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # # 10
                if A == 1:
                    plt.figure()
                    plt.imshow(im_Binary_OTSU)
                    plt.show()
                _N_M_S_A_features = Features_N_M_S_A (im_Binary_OTSU)  #<<<<<<<<<<<<<<<<<<
                for f in _N_M_S_A_features:
                    FINAL_FEATURES.append(f)


                if 1==1:
                    Lines_list =  Lines(im_Gray)
                    for f in Lines_list:
                        FINAL_FEATURES.append(f)
                ###T_FINAL_FEATURES.append(FINAL_FEATURES)
                #FINAL_FEATURES = np.array(FINAL_FEATURES)

                FINAL_FEATURES = np.array(FINAL_FEATURES)
                FINAL_FEATURES = FINAL_FEATURES.reshape(1,-1)
                return FINAL_FEATURES

                    # im_Binary_GAUSSIAN_3 = np.zeros((160,160,3))
                    # im_Binary_GAUSSIAN_3[:,:,0],im_Binary_GAUSSIAN_3[:,:,1], im_Binary_GAUSSIAN_3[:,:,2]  = im_Binary_GAUSSIAN,im_Binary_GAUSSIAN,im_Binary_GAUSSIAN
                    # plt.figure()
                    # plt.imshow(im_Binary_GAUSSIAN_3)
                    # plt.show()




#________________ PXM __________________________________________
def HC2_Feature_Extractor (divider,width,heigth,n_channels,paths):
    #depth = int(300/divider)
    depth = int(80/divider)
    FEATURES = np.zeros(((paths.shape[0]), depth, 1621))  # 120  114
    
    ###FEATURES = np.zeros(((paths.shape[0]),  depth, CNN2D_FE.output_shape[1]))
    index = -1
    for path in paths:
        index+=1
        path = path.reshape(-1,1)

        # DeepFake_________
        # Compression Time ____
        video = video_loading (path,heigth, width) #resize
        video = Video_Subasampling (video, divider)#subsamble
        # Compression Time ____

        # Pneymonia_________
        # image_3d = data_3d_loading(path, divider, heigth, width)
 
        image_3d = video
        # FE Time ____
        F = np.zeros((image_3d.shape[0],1621))  #257 
        ind = -1   # 41, 43
        for img in image_3d:
            ind+=1
            # if ind == 44:
            #     s = 1
            f1 = Other_Features (img) 
            #f2 = Co_Oc_Features (img) 
            f = f1#np.concatenate((f1,f2),axis=1)
            F[ind] = f

        FEATURES[index] = F

    return FEATURES



#________________ DFT __________________________________________

# Durall et al:
# Exposing Deepfakes using simble features_____________________
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
# Exposing Deepfakes using simble features_____________________

def DFT(IMAGES):
        DFT_Azimouthial = []
        for b_im in [IMAGES]:
            b_im_inv = np.copy (b_im)
            a0, a1, a2 = b_im[:,:,0], b_im[:,:,1], b_im[:,:,2]
            f0, f1, f2 = np.fft.fft2(a0), np.fft.fft2(a1), np.fft.fft2(a2)
            fshift0, fshift1, fshift2 = np.fft.fftshift(f0), np.fft.fftshift(f1), np.fft.fftshift(f2)

            magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2 = 20*np.log(np.abs(fshift0)), 20*np.log(np.abs(fshift1)), 20*np.log(np.abs(fshift2))

            b_im_inv[:,:,0], b_im_inv[:,:,1], b_im_inv[:,:,2] = magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2

            a0, a1, a2 = b_im_inv[:,:,0], b_im_inv[:,:,1], b_im_inv[:,:,2]

            mean = []
            for __im in [magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2]:
                m = list(azimuthalAverage(__im, center=None))
                mean = m + mean
            DFT_Azimouthial.append(mean)
            
        DFT_Azimouthial = np.array(DFT_Azimouthial)
        ###########DFT_Azimouthial = Standarize (DFT_Azimouthial)
        return DFT_Azimouthial


def HC1_Feature_Extractor (divider,width,heigth,n_channels,paths):
    depth = int(80/divider)
    FEATURES = np.zeros(((paths.shape[0]), depth, 99))  # 333
    
    ###FEATURES = np.zeros(((paths.shape[0]),  depth, CNN2D_FE.output_shape[1]))
    index = -1
    for path in paths:
        index+=1
        path = path.reshape(-1,1)

        # DeepFake ____
        ##video = video_loading (path,heigth, width) #resize
        ##video = Video_Subasampling (video, divider)#subsamble

        # Pneymonia_________
        video = data_3d_loading(path, divider, heigth, width)


        # FE Time ____
        F = np.zeros((video.shape[0],99))  # 333
        ind = -1
        for img in video:
            ind+=1
            f1 = DFT(img)
            #f2 = Co_Oc_Features (img) 
            f = f1#np.concatenate((f1,f2),axis=1)
            F[ind] = f

        FEATURES[index] = F

    return FEATURES





#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________



# ______________PCA
#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________



# from sklearn.decomposition import PCA



# pca = PCA(n_components=5160)
# F_Patch = pca.fit(F_Patch).transform(F_Patch)






#_______________________________________________________________________________________________________________________________________________________________
#_____________ CNN Compile and Train _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________


#import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Compatible with tensorflow backend
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed


def Compile_CNN(model,lr):
                    # Compile model.
                    initial_learning_rate = lr 
                    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
                    )
                    model.compile(
                        loss='binary_crossentropy',#focal_loss(),  # 'binary_crossentropy'   'categorical_crossentropy', 'sparse_categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),   # 0.001
                        metrics=['mean_squared_error'] # Geometric_Mean
                    )

                    # Define callbacks.
                    checkpoint_cb = keras.callbacks.ModelCheckpoint(
                        "model.epoch{epoch:02d}-loss{val_mean_squared_error:.4f}.h5",
                        mode='min',
                        monitor='val_mean_squared_error', # Geometric_Mean
                        save_best_only=True
                    )
                    early_stopping_cb = keras.callbacks.EarlyStopping(monitor=['val_acc'], patience=3)
                    return model, checkpoint_cb, early_stopping_cb


def Train_CNN(model, epochs, train_video_g, test_video_g, checkpoint_cb, early_stopping_cb):
                    model.fit(train_video_g,
                                                    verbose=1,
                                                    epochs=epochs,
                                                    #############################callbacks=[checkpoint_cb, early_stopping_cb],
                                                    #validation_data=test_video_g,
                                                    shuffle=False)
                    return model

def Train_CNN_2(model, epochs, train_video,tr_labels, test_video,ts_labels, checkpoint_cb, early_stopping_cb):
                    model.fit(train_video,tr_labels,
                                                    verbose=1,
                                                    epochs=epochs,
                                                    callbacks=[checkpoint_cb],
                                                    validation_data=(test_video,ts_labels),
                                                    shuffle=False)
                    return model



def Train_CAE_CNN(cnn_classifier, cnn_encoder, epochs, divider, PATHS, LABELS, checkpoint_cb, early_stopping_cb):

    from random import shuffle


    Bbatch_size = 32 # THIS IS THE bs FOR THE CNN CLASIFIER, since the AE TAKES THE HUGE INITIAL INPUT, WE USE THERE bs = 1 to avoid memory error
                     # N = int (len(train_paths) / batch_size) 
    for _ in range (epochs):
            if 1==1:
                ind_list = [_r_ for _r_ in range(len(PATHS))]
                shuffle(ind_list)
                PATHS  = PATHS[ind_list]
                LABELS  = LABELS[ind_list]

            probs = []
            X = np.zeros((Bbatch_size, 10, 160,160,3))#cnn_encoder.output_shape[1], cnn_encoder.output_shape[2], cnn_encoder.output_shape[3], cnn_encoder.output_shape[4]))
            Y = np.zeros((Bbatch_size, LABELS.shape[1]))
            i = 0
            cnt = 0
            for path, y in zip (PATHS, LABELS):
                i+=1
                cnt += 1
                path = path.reshape(-1,1)

                x = video_loading (path)
                x = Video_Subasampling (x, divider)

                x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2],x.shape[3])
                y = y.reshape(1,LABELS.shape[1])

                if 1==2:
                    # Store AE Outputs
                    vid_path = path[0][0]
                    _,__,name = vid_path.split('/')
                    os.mkdir('AE_Models/Test_AE_3D/AE_10_160_160/'+name)
                    _cnt__ = 0
                    for fr_s in x:
                        _cnt__+=1
                        skimage.io.imsave ('AE_Models/Test_AE_3D/AE_10_160_160/'+name+'/'+str(_cnt__)+'.jpg',fr_s)   
                    
                # Train    
                if 1==1:
                    x = cnn_encoder.predict(x)[0]

                    x = resize(x, (10, 160,160,3))

                    X[i-1] = x
                    Y[i-1] = y
                    if i % Bbatch_size == 0 or cnt == len(PATHS):
                        cnn_classifier.fit(X, Y, batch_size = 16,
                                                            verbose=1,
                                                            #############################callbacks=[checkpoint_cb, early_stopping_cb],
                                                            #validation_data=test_video_g,
                                                            shuffle=False)
                        i = 0
                        
            #             prb = cnn_classifier.predict(X) 
            #             prb = list(prb)
            #             for pr in prb:
            #                 probs.append(pr)
            # probs = np.array(probs)
            # probs = probs[:len(train_paths)]  
 
    return cnn_classifier , probs


def Predict_CAE_CNN (cnn_classifier, cnn_encoder, divider, test_paths):

    Bbatch_size = 32 # THIS IS THE bs FOR THE CNN CLASIFIER,  since the AE TAKES THE HUGE INITIAL INPUT,  WE USE THERE bs = 1 to avoid memory error N = int (len(train_paths) / batch_size) 

    X = np.zeros((Bbatch_size, 10, 160,160,3))#cnn_encoder.output_shape[1], cnn_encoder.output_shape[2], cnn_encoder.output_shape[3], cnn_encoder.output_shape[4]))
    probs = []
    i = 0
    cnt = 0
    for path in  test_paths:
                i+=1
                cnt+=1
                path = path.reshape(-1,1)
                x = video_loading (path)
                x = Video_Subasampling (x, divider)

                
                x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2],x.shape[3])

                x = cnn_encoder.predict(x) [0]
                x = resize(x, (10, 160,160,3))


                X[i-1] = x

                if i % Bbatch_size == 0 or cnt == len(test_paths):
                    i = 0
                    prb = cnn_classifier.predict(X) 
                    prb = list(prb)
                    for pr in prb:
                        probs.append(pr)

    probs = np.array(probs)
    probs = probs[:len(test_paths)]                       
    return probs




#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________










#_______________________________________________________________________________________________________________________________________________________________
#_____________ General_Functions _________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________

def LOOCV (ratio, f_train, l_train, c, it):
            # ratio = 0.001, 0.01, 0.1
            if int (f_train.shape[0]*ratio) == 0: SPLITS = f_train.shape[0]
            else:
                SPLITS = int (f_train.shape[0] / int (f_train.shape[0]*ratio)) 
            kf = KFold(n_splits=SPLITS)
            kf.get_n_splits(f_train)

            Total_Labels = []
            Total_Preds = []

            for train_index, vl_index in kf.split(f_train):
                f_tr_, f_vl_ = f_train[train_index], f_train[vl_index]
                tr_labels_, vl_labels_ = l_train[train_index], l_train[vl_index]
                __model = LogisticRegression(C=c, max_iter=it).fit(f_tr_, tr_labels_)
                Total_Labels.append(vl_labels_) 
                Total_Preds.append(__model.predict(f_vl_))

            Total_Preds__, Total_Labels__ = [],[]
            for PR,L in zip (Total_Preds, Total_Labels):
                for pr,l in zip(PR,L):
                    Total_Preds__.append(pr)
                    Total_Labels__.append(l)
            Total_Preds__, Total_Labels__ = np.array(Total_Preds__), np.array(Total_Labels__)
            gm = Evaluation(Total_Preds__, Total_Labels__)
            return gm


def std_under_mean (arr):
    mean = np.mean(arr)
    arr_under_mean = arr[:,np.argwhere (arr<=mean)[:,1]]
    std_under = np.std(arr_under_mean)
    return std_under




def LOOCV_Features_Attribution (ratio, f_train, l_train, c, it):

            if int (f_train.shape[0]*ratio) == 0: SPLITS = f_train.shape[0]
            else:
                SPLITS = int (f_train.shape[0] / int (f_train.shape[0]*ratio)) 
            kf = KFold(n_splits=SPLITS)
            kf.get_n_splits(f_train)

            COEF_real = np.zeros((1,f_train.shape[1]))
            for train_index, vl_index in kf.split(f_train):
                f_tr_ = f_train[train_index]
                tr_labels_ = l_train[train_index]
                __model = LogisticRegression(C=c, max_iter=it).fit(f_tr_, tr_labels_)

                coef = __model.coef_
                COEF_real = coef+COEF_real

            return COEF_real



def LOOCV_Feature_Reduction (COEF_abs, f_train, l_train, c, it):

            mean, std = np.mean(COEF_abs), np.std(COEF_abs)
            std_under = std_under_mean(COEF_abs)
            GM_VAL, IDS = [], []
            IDS.append(np.argwhere(COEF_abs >= 0 )[:,1])
            gm = LOOCV (0.01, f_train, l_train, c, it)  # 0.01   0.001
            GM_VAL.append(gm)
            while  1==1:
                #ids = np.argwhere(COEF_abs >= mean - 0.99*std )[:,1]
                ids = np.argwhere(COEF_abs >= mean-std_under)[:,1]
                IDS.append(ids)
                if (len(IDS[-2])-len(IDS[-1])) == 0 :
                        break
                f_train_r = f_train[:,ids]
                COEF_abs_r = COEF_abs[0,ids].reshape(1,-1)
                mean, std = np.mean(COEF_abs_r), np.std(COEF_abs_r)
                std_under = std_under_mean(COEF_abs_r)

                gm = LOOCV (0.01, f_train_r, l_train, c, it) # 0.01   0.001
                GM_VAL.append(gm)

            id_gm_val_max = np.argwhere(GM_VAL==np.max(GM_VAL))   
            ids_max = IDS[id_gm_val_max[0][0]] 

            return ids_max, np.max(GM_VAL), IDS, GM_VAL



def LOOCV_PeCo_Feature_Reduction (f_train, l_train, c, it):

            init_shape = f_train.shape[1]
            GM_VAL, IDS = [], []
            IDS.append([i for i in range (f_train.shape[1])])
            gm = LOOCV (0.01, f_train, l_train, c, it)  # 0.01   0.001
            GM_VAL.append(gm)
            e = 0
            f_train = pd.DataFrame(f_train)

            while  1==1:

                corr = f_train.corr()
                columns = np.full((corr.shape[0],), True, dtype=bool)
                # ____ pearson corelation _____
                for i in range(corr.shape[0]):
                    for j in range(i+1, corr.shape[0]):
                        if corr.iloc[i,j] >= 0.95 - e:
                            if columns[j]:
                                columns[j] = False
                selected_columns = f_train.columns[columns]
                f_train = f_train[selected_columns]
                s = 1

                IDS.append(selected_columns.values)
                if (len(IDS[-2])-len(IDS[-1])) == 0 :
                        gm = LOOCV (0.01, f_train.values, l_train, c, it) # 0.01   0.001
                        GM_VAL.append(gm)
                        break

                # if (len(IDS[-2]) <= init_shape / 4)  :
                #         break

                gm = LOOCV (0.01, f_train.values, l_train, c, it) # 0.01   0.001
                GM_VAL.append(gm)
                

                #e = e+0.01

            id_gm_val_max = np.argwhere(GM_VAL==np.max(GM_VAL))   
            ids_max = IDS[id_gm_val_max[0][0]] 

            return ids_max, np.max(GM_VAL), IDS, GM_VAL






def LOOCV_Feature_Searching (ids_max, gm_max, COEF_abs, f_train, l_train, c, it):

            GM_VAL = np.zeros((100))
            IDS_R = np.ones((100)) * (-1)
            IDS_R = IDS_R.astype(int)
            GM_VAL[0] = LOOCV (0.01, f_train[:,ids_max], l_train, c, it)
            feat_dim = 1

            COEF_max = COEF_abs[:,ids_max]
            _ = np.argsort(COEF_max).reshape(-1,)
            ids_max_sorted = ids_max[_]
            
            splt = int(len(ids_max_sorted)/5)
            id_v_low, id_low, id_aver, id_high, id_v_high = np.copy(ids_max_sorted[:splt]), np.copy(ids_max_sorted[splt:splt*2]), np.copy(ids_max_sorted[splt*2:splt*3]), np.copy(ids_max_sorted[splt*3:splt*4]), np.copy(ids_max_sorted[splt*4:])
            for ID in [id_v_low, id_low, id_aver, id_high, id_v_high]:
                while 1==1:   
                    for id in ID: 
                        if id in ids_max_sorted:
                            i = np.argwhere(id==ids_max_sorted)[0][0]
                            f_train_r = f_train[:,list(ids_max_sorted [:i]) + list(ids_max_sorted [i+1:])]
                            gm = LOOCV (0.01, f_train_r, l_train, c, it)

                            if GM_VAL[feat_dim] <gm :
                                GM_VAL[feat_dim] = gm
                                IDS_R[feat_dim] = i
     
                    if GM_VAL[feat_dim-1]-GM_VAL[feat_dim-2]<0:
                        break
                    else:
                        ID = list(ID [:IDS_R[feat_dim]]) + list(ID [IDS_R[feat_dim]+1:])
                        ids_max_sorted = list(ids_max_sorted [:IDS_R[feat_dim]]) + list(ids_max_sorted [IDS_R[feat_dim]+1:])
                        feat_dim+=1

            s = 1
            return ids_max_sorted



def Feature_Optimization (tr_data, tr_labels, c, it):

    GM_VAL = np.zeros((100))
    f_tr_con = [] # tr_data[:,(65, 73, 97, 230, 231, 278, 74, 172, 173, 69)] # []
    F_IDS = np.ones((100)) * (-1)
    feat_dim = 0
    while 1==1:

        for i in range(tr_data.shape[1]):
            if i not in F_IDS:
                
                f_tr = tr_data[:,i].reshape(-1,1)
                if len(f_tr_con)>0:
                    f_tr = np.concatenate((f_tr_con,f_tr),axis=1)
                gm = 0
                kf = KFold(n_splits=int (tr_data.shape[0]/5))  # 100
                kf.get_n_splits(f_tr)
                Total_Labels = []
                Total_Preds = []
                for train_index, vl_index in kf.split(f_tr):

                    f_tr_, f_vl_ = f_tr[train_index], f_tr[vl_index]
                    tr_labels_, vl_labels_ = tr_labels[train_index], tr_labels[vl_index]

     
                    tr_labels_ = tr_labels_.ravel()
                    __model = LogisticRegression(C=c, max_iter=it).fit(f_tr_, tr_labels_)

                    Total_Labels.append(vl_labels_)
                    Total_Preds.append(__model.predict(f_vl_))

                Total_Preds__, Total_Labels__ = [],[]
                for PR,L in zip (Total_Preds, Total_Labels):
                    for pr,l in zip(PR,L):
                        Total_Preds__.append(pr)
                        Total_Labels__.append(l)
                Total_Preds__, Total_Labels__ = np.array(Total_Preds__), np.array(Total_Labels__)
                gm = Evaluation(Total_Preds__, Total_Labels__)

                if GM_VAL[feat_dim] < gm:   # 99.0, 773.0, 550.0, 410.0, 824.0, 629, 246, 453, 469,
                    GM_VAL[feat_dim] = gm
                    F_IDS [feat_dim] = i
                    max_f_tr  = f_tr[:,-1].reshape(-1,1)

                    s = 1 

        feat_dim += 1
        if len(f_tr_con)>0:
            f_tr_con  = np.concatenate((max_f_tr,f_tr_con),axis=1)
        else:
            f_tr_con = max_f_tr

        # Evaluation(__model.predict(vl_data[:,(2,35)]), vl_labels)
        # if GM_VAL[-2] >= GM_VAL[-1]*0.99:
        #     break
    return F_IDS




# 




def Standarize (data):
    fff = np.copy(data)
    # # # #---------------------------> Standarize
    for i in range(len(fff[0])):
                fff[:, i] = (fff[:, i] - np.mean(fff[:,i]))/np.std(fff[:,i])
    ii = np.argwhere(np.isnan(fff))
    for i,j in ii:
            fff[i,j] = 0

    return fff


def video_loading (paths,heigth, width):
        index = -1
        for vid_path in paths:  
                    index+=1
                    vid_path = vid_path[0]
                    #_,__,name = vid_path.split('/')
                    ###_,__,___,name = vid_path.split('/')

                    Frames_id = os.listdir(vid_path)

                    Frames_id_int = []
                    for fr in Frames_id:
                            if '_' not in fr:    
                                        fr = int (fr.replace('.jpg',''))
                                        Frames_id_int.append (fr)
                    Frames_id_int = np.array(Frames_id_int)
                    Frames_id_int = np.sort(Frames_id_int) 

                    video = []
                    for fr_int in Frames_id_int:
                                    fr_str = str(fr_int)+'.jpg'
                                    img_path = vid_path+'/'+fr_str
                                    img = io.imread(img_path) # all frames must be of same size
                                    img = img.astype(float)
                                    img = cv2.resize(img, (heigth, width))
                                    video.append(img)  
                    video = np.array(video)
        return video.astype(float)/255.

def Video_Subasampling (video, divider):
                        video_sampled = []
                        init_depth = video.shape[0]
                        #divider = int (init_depth/depth)
                        for _i in range(init_depth):
                                if _i % divider == 0:
                                    video_sampled.append(video[_i])
                        video_sampled = np.array(video_sampled)

                        return video_sampled.astype(float)

def decode_labels (labels):
    int_labels = []
    for lb in labels:
        if lb == 'FAKE':
            int_labels.append(1)
        else:
            int_labels.append(0)
    int_labels = np.array(int_labels)
    return int_labels

def create_paths (folder_name,filenames):
    paths = []
    for f in filenames:
        f,_ = f.split('.mp4')
        path = folder_name+f
        paths.append(path)
    paths = pd.DataFrame(paths).values
    return  paths



def Balance_Data (X,Y):

    cnt_label1, cnt_label0 = 0, 0
    X0,X1 = [],[]
    for x,y in zip (X,Y):
        if y == 1:
            cnt_label1+=1
            X1.append(x)
        else:
            cnt_label0+=1
            X0.append(x)

    if cnt_label0>cnt_label1:
        final_N = cnt_label1
    else:
        final_N = cnt_label0

    s1 = int(cnt_label1 / final_N)
    s0 = int(cnt_label0 / final_N)

    X1_b = []
    for i in range (cnt_label1):
        if i % s1  == 0:
            X1_b.append(X1[i])
    X0_b = []
    for i in range (cnt_label0):
        if i % s0  == 0:
            X0_b.append(X0[i])
    X0_b = np.array(X0_b)
    X1_b = np.array(X1_b)
    Y0_b = np.zeros((X0_b.shape[0],1))
    Y1_b = np.ones((X1_b.shape[0],1))

    X = np.concatenate((X0_b, X1_b),axis=0)
    Y = np.concatenate((Y0_b, Y1_b),axis=0)
    return X,Y


###from sklearn.neural_network import MLPClassifier
##MLPClassifier(random_state=0, solver='lbfgs', hidden_layer_sizes=(128, 16)).fit(f_train, Labels_train)


def Evaluation(preds, labels):
    
# tn, fp, fn, tp = 18, 3, 48, 37
# total = tn + fp + fn + tp
# tn, fp, fn, tp = 18, 3, 48, 37
# if (tn + fp + fn + tp) == total:
    # print ('Acc = ')
    # print(str(np.round ((tp+tn)/(tp+tn+fp+fn), 3)*100))
    # print ('GM = ')
    # print(np.round (((tp*tn)**(0.5)) / (((tp+fn)*(tn+fp))**(0.5)), 3))
    # print ('Sen = ')
    # print(np.round (tp / (tp + fn), 3))
    # print ('Spe = ')
    # print(np.round (tn / (tn+fp), 3))
    # print (tn, fp, fn, tp)


                    test_y = labels
                    pr_t   = preds

                    Acc = accuracy_score(test_y, pr_t) 
                    F1 = f1_score(test_y, pr_t)
                    tn, fp, fn, tp = confusion_matrix(test_y, pr_t).ravel()


                    GM = ((tp*tn)**(0.5)) / (((tp+fn)*(tn+fp))**(0.5))
                    sensitivity =  tp / (tp + fn) # sensitivity, recall
                    specificity = tn / (tn+fp) # specificity, selectivity
                    #ROC_AUC = metrics.roc_auc_score(test_y, probs)  # [:,1]


                    print ('CM = ')
                    print(np.round (confusion_matrix(test_y, preds), 3))                    
                    print ('Acc = ')
                    print(str(np.round (Acc, 3)*100))
                    print ('GM = ')
                    print(np.round (GM, 3))
                    # print ('F1 = ')
                    # print(np.round (F1, 3))
                    print ('Sen = ')
                    print(np.round (sensitivity, 3))
                    print ('Spe = ')
                    print(np.round (specificity, 3))
                    # print ('ROC_AUC = ')
                    # print(np.round (ROC_AUC, 3))

                    return GM



def Evaluation_Multi(preds, labels):

                    test_y = labels
                    pr_t = preds

                    Acc = accuracy_score(test_y, pr_t) 
                    #F1 = f1_score(test_y, pr_t)
                    #tn, fp, fn, tp = confusion_matrix(test_y, pr_t).ravel()

                    CM = confusion_matrix(test_y, pr_t)
                    Diag, Other  = [],[]
                    for i in range (CM.shape[0]):
                        Diag .append(CM[i,i]) 
                        Other .append(np.sum(CM[i,:i])+np.sum(CM[i,i+1:])) 


                    print(CM)
                    print(Acc)

                    return Diag, Other





def Create_2D_Dataset (depth,width,heigth,n_channels,train_paths,Labels_train):
    for vid_path, lbl in zip(train_paths,Labels_train):  
                vid_path = vid_path[0]
                _,__,name = vid_path.split('/')
                Frames_id = os.listdir(vid_path)

                Frames_id_int = []
                for fr in Frames_id:
                        if '_' not in fr:    
                                    fr = int (fr.replace('.jpg',''))
                                    Frames_id_int.append (fr)
                Frames_id_int = np.array(Frames_id_int)
                Frames_id_int = np.sort(Frames_id_int) 

                Frames_per_Instance = []
                for fr_int in Frames_id_int:
                                fr_str = str(fr_int)+'.jpg'
                                img_path = vid_path+'/'+fr_str
                                img = io.imread(img_path) # all frames must be of same size
                                img = img.astype(float)
                                Frames_per_Instance.append(img)  
                Frames_per_Instance = np.array(Frames_per_Instance)

                Frames_per_Instance_Sampled = np.zeros((depth,width,heigth,n_channels))
                init_depth = Frames_per_Instance.shape[0]
                sampling_Ratio = int (init_depth/depth)
                j=-1
                for _i in range(300):
                        if _i % sampling_Ratio == 0:
                            j+=1
                            Frames_per_Instance_Sampled[j] = cv2.resize(Frames_per_Instance[_i],(width,heigth))#.astype(float)/255.
                            # plt.figure()
                            # plt.imshow(Frames_per_Instance_Sampled[j])
                            # plt.show()

                _cnt = -1
                for fr_s in Frames_per_Instance_Sampled:
                    _cnt+=1
                    skimage.io.imsave ('Dataset/Test_Dataset_2D/'+str(int(lbl))+'/'+str(_cnt)+name+'.jpg',fr_s)    
                    # Train_Dataset_2D   # lbl[0]

#_______________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________________









# flow_from_directory(
#         'data/train',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')






























if 1==2:
    divider = 2
    heigth, width = 300,300
    depth = 80/divider

    # for vid_path in train_paths:  
    #         image_3d = data_3d_loading(vid_path, divider, heigth, width)





    CAE = Build_AE3D_EE(depth, width, heigth)
    CAE.summary()
    #CAE = load_model('../input/ddddddddd/CAE_50.h5')

    train_g = Data_Generator_3D_CNN(mode='train_AE', batch_size=1, divider=divider, width=width, heigth=heigth, paths = train_paths, labels = Labels_train_oh, n_channels=3, shuffle=False)


    lr = 0.001    # 0.001  0.0005   0.0001  0.00005
    CAE, checkpoint_cb, early_stopping_cb = Compile_CNN (CAE, lr)
        # Training  (Very Slow)
    if 1==1:    
                    epochs = 5
                    CAE = Train_CNN(CAE, epochs, train_g, train_g, checkpoint_cb, early_stopping_cb)  
    
    s = 1
