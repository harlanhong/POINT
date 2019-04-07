import os
import numpy as np
import random

import skimage
from skimage import data, io
# import skimage.io
import skimage.transform
import skimage.color

import pickle
import fnmatch

import scipy.io as scio
import pdb
DataPath = './'
DataName1 = 'indexAndEventLabel.mat'
data = scio.loadmat(DataName1)
index_train = data['index_train']
eventLabel_train=data['eventLabel_train']
index_val = data['index_val']
eventLabel_val=data['eventLabel_val']
index_test = data['index_test']
eventLabel_test=data['eventLabel_test']
# pdb.set_trace()
train_img_id=[]
train_event_id=[]
val_img_id=[]
val_event_id=[]
test_img_id=[]
test_event_id=[]
for i in range(index_train.shape[1]):
    train_img_id.append(index_train[0,i])
    train_event_id.append(eventLabel_train[i,0])
for i in range(index_val.shape[1]):
    val_img_id.append(index_val[0,i])
    val_event_id.append(eventLabel_val[i,0])
for i in range(index_test.shape[1]):
    test_img_id.append(index_test[0,i])
    test_event_id.append(eventLabel_test[i,0])

np.save(DataPath+ 'eventLabel.npy', {'index_train': train_img_id, 'index_val': val_img_id, 'index_test': test_img_id,
                                     'eventLabel_train':train_event_id,'eventLabel_val':val_event_id,'eventLabel_test':test_event_id})

