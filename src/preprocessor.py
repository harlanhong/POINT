from __future__ import absolute_import
import os.path as osp
import pdb
from PIL import Image
import torch
import numpy as np

class ExteroiPreprocessor(object):
    def __init__(self, dataset, isTrain=True, transform1=None, transform2=None,transform3=None):
        # super(Preprocessor, self).__init__()
        self.dataset = dataset
  
        self.transform1 = transform1
        self.isTrain = isTrain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        image,Imgid= self.dataset[index]
        srcImg = Image.open(image).convert('RGB')
        srcImg = np.asarray(srcImg)
        # if self.transform1 is not None:
        #     srcImg = self.transform1(srcImg)
        return srcImg,Imgid
#relation network
class RelationNetCPreprocessor(object):
    def __init__(self, dataset, isTrain=True,  transform1=None, transform2=None,transform3=None):
        # super(Preprocessor, self).__init__()
        self.dataset = dataset
     
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.isTrain = isTrain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):

        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        src,Face, FaceCont,Coor, label, ImgId = self.dataset[index]
        srcImg = Image.open(src).convert('RGB')
        FaceImg = Image.open(Face).convert('RGB')
        FaceContImg = Image.open(FaceCont).convert('RGB')
        Coor = Image.open(Coor).convert('L')
        Coor = np.array(Coor,dtype=np.float32)
        # srcImg=np.array(srcImg,dtype=np.float32)
        # img = Image.open(frame_dir).convert('RGB')
        if self.transform1 is not None:
            Face = self.transform1(FaceImg)
        if self.transform2 is not None:
            FaceCont = self.transform2(FaceContImg)
        if self.transform3 is not None:
            src = self.transform3(srcImg)
        return src,Face,FaceCont,Coor,label, ImgId
