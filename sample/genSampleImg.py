import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import keras
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

import random
import numpy as np



def genSampleImg(images,labels,randomize=True):
    X_img=np.empty((images.shape[0]*28,28,28))
    Y_lab=np.empty(images.shape[0]*28)
    for i in range(images.shape[0]):
        if randomize==True:
            seq1=np.random.choice(np.arange(28),28,replace=False)
        else:
            seq1=np.arange(28)
        nan_array=np.empty((28,28))
        nan_array[:]=np.nan
        for j in range(28):
            nan_array[seq1[j],:]=1
            img=images[i]*nan_array
            X_img[i*28+j]=img
            Y_lab[i*28+j]=labels[i]
    return(X_img,Y_lab)

sample=np.random.choice(np.arange(X_train.shape[0]),100)
R1=genSampleImg(X_train[sample],Y_train[sample])
R2=genSampleImg(X_train[sample],Y_train[sample],randomize=False)

import pickle
with open('RandomImg.pickle', 'wb') as f:
    pickle.dump(R1, f)

with open('Sequential.pickle', 'wb') as f:
    pickle.dump(R2, f)

