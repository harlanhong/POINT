from __future__ import print_function, absolute_import
import os.path as osp

from collections import defaultdict

import numpy as np
import pdb
from torch.utils.data import Dataset
import scipy.io as scio
from glob import glob
import pdb


# from .dataset import Dataset
# from ..utils.osutils import mkdir_if_missing
# from ..utils.serialization import write_json


class RelatinNetCMSIP(Dataset): #img+face+faceCont+coor
    # url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    # md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root,index='./MSindex.npy'):
        # processing Train
        # self.num_videos = 8
        self.root = root
        index = np.load(index).tolist();
        index_train = index['train']
        index_test = index['test']
        index_val = index['val']
        # A list of Image Folders
        ImageFolderList = sorted(glob(root + '/Image_*'))
        self.train = []
        self.num_train = len(index_train)
        for i in range(self.num_train):
            ind = index_train[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_)-5]
                self.train.append((src[j],Faces[j], FaceConts[j], Coor[j],int(Label_), ind))
            if NumFace == 1:
                self.train.append((src[0],Faces[0], FaceConts[0],Coor[0], 0, ind))


        # testing data
        self.test = []
        self.num_test = len(index_test)
        for i in range(self.num_test):
            ind = index_test[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_) - 5]
                self.test.append((src[j],Faces[j], FaceConts[j],Coor[j], int(Label_), ind))


        # val data
        self.val = []
        self.num_val = len(index_val)
        for i in range(self.num_val):
            ind = index_val[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_) - 5]
                self.val.append((src[j],Faces[j], FaceConts[j],Coor[j], int(Label_), ind))
