from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

#src,Face, FaceCont,Coor, label, ImgId
class CRandomFaceSampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.Img_dic = defaultdict(list)
        self.Imp_dic = defaultdict(list)
        for index, (src,Face, FaceCont,Coor, label, ImgId) in enumerate(data_source):
            # for ind, frame in enumerate(frames):
            if label == 0:
                self.Img_dic[ImgId].append(index)  #[Img_dic[ImgId] is a list which contains the sample index of ImagId]
            if label == 1:
                self.Imp_dic[ImgId].append(index)
        self.ImgId = list(self.Img_dic.keys())
        self.num_samples = len(self.ImgId)
        # self.sample()

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            ImgId = self.ImgId[i]
            # *************select important face*************
            ImpId = self.Imp_dic[ImgId]
            if len(ImpId)==0:
                continue
            ImpId = np.random.choice(ImpId, size=1,replace=False)

            ret.extend(ImpId)
            # ***********select non-important face***********
            ImpId = self.Img_dic[ImgId]
            if len(ImpId) >= self.num_instances:
                ImpId = np.random.choice(ImpId, size=self.num_instances-1, replace=False)
            else:
                ImpId = np.random.choice(ImpId, size=self.num_instances-1, replace=True)
            ret.extend(ImpId)
        self.ret = ret
        return iter(ret)
class CTestSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.Img_dic = defaultdict(list)
        for index, (src,Face, FaceCont,Coor, label, ImgId) in enumerate(data_source):
            # for ind, frame in enumerate(frames):
            self.Img_dic[ImgId].append(index)
        self.ImgId = list(self.Img_dic.keys())
        self.num_samples = len(self.ImgId)
        # self.sample()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            ImgId = self.ImgId[i]
            # *************select important face*************
            index = self.Img_dic[ImgId]
            ret.extend(index)
        self.ret = ret
        return iter(ret)
#src,Face, FaceCont,Coor,faceRoi,faceConRoi label, ImgId
class RelationNetRandomFaceSampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.Img_dic = defaultdict(list)
        self.Imp_dic = defaultdict(list)
        for index, (src,Face, FaceCont,Coor,label, ImgId) in enumerate(data_source):
            # for ind, frame in enumerate(frames):
            if label == 0:
                self.Img_dic[ImgId].append(index)  #[Img_dic[ImgId] is a list which contains the sample index of ImagId]
            if label == 1:
                self.Imp_dic[ImgId].append(index)
        self.ImgId = list(self.Img_dic.keys())
        self.num_samples = len(self.ImgId)
        # self.sample()

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            ImgId = self.ImgId[i]
            # *************select important face*************
            ImpId = self.Imp_dic[ImgId]
            if len(ImpId)==0:
                continue
            ImpId = np.random.choice(ImpId, size=1,replace=False)
            ret.extend(ImpId)
            # ***********select non-important face***********
            ImpId = self.Img_dic[ImgId]
            if len(ImpId) >= self.num_instances:
                ImpId = np.random.choice(ImpId, size=self.num_instances-1, replace=False)
            else:
                ImpId = np.random.choice(ImpId, size=self.num_instances-1, replace=True)
            ret.extend(ImpId)
        self.ret = ret
        return iter(ret)
class RelationNetTestSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.Img_dic = defaultdict(list)
        for index, (src,Face, FaceCont,Coor, label, ImgId) in enumerate(data_source):
            # for ind, frame in enumerate(frames):
            self.Img_dic[ImgId].append(index)
        self.ImgId = list(self.Img_dic.keys())
        self.num_samples = len(self.ImgId)
        # self.sample()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # indices = torch.randperm(self.num_samples)
        indices = torch.IntTensor(np.arange(0,self.num_samples,1,np.int32))
        ret = []
        for i in indices:
            ImgId = self.ImgId[i]
            # *************select important face*************
            index = self.Img_dic[ImgId]
            ret.extend(index)
        self.ret = ret
        return iter(ret)