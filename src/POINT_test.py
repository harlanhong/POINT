from __future__ import print_function

import argparse
import copy
# from vps import VPS
import pdb
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms as T

from dataset import *
from GetMapAndCMC import *
# from sampler import RandomIdentitySampler
from preprocessor import *
from relationNetwork import *
from sampler import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset-root', type=str, default='./',
                    help='directory of all images')
parser.add_argument('--nw', type=int, default=12,
                    help='number of workers')
parser.add_argument('--ni', type=int, default=2,
                    help='number of instances for each tracklet (default is 2)')
parser.add_argument('--tl', type=int, default=2,
                    help='number of tracklets for each identity (default is 2)')
parser.add_argument('--gpu-id', type=int, default=-1,
                    help='gpu id (default is 2)')
parser.add_argument('--log-dir', type=str, default='./',
                    help='directory of the log file (default is ./)')

parser.add_argument('--model',type=str,default='',help='model name')
parser.add_argument('--h',type=int, default=4,help='the number of relation submodule')
parser.add_argument('--N',type=int,default=2,help='the number of relation module')
parser.add_argument('--index_name',type=str,default='../data/MSindex.npy',help='the name of index of train, val and test set')
parser.add_argument('--dataset_path',type=str,default='../data/MSDataSet_process',help='the path of dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


VIPDataset = RelatinNetCMSIP(root=args.dataset_path,index=args.index_name)


normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
        # T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        normalizer
        ])

test_set = VIPDataset.test
test_num = VIPDataset.num_test

test_loader = DataLoader(
    RelationNetCPreprocessor(test_set, isTrain=False,
                     transform1=test_transformer,transform2=test_transformer,transform3=test_transformer),
                     sampler = RelationNetTestSampler(data_source=test_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)
print(type(test_set[0][0]))
relation=1
modelName = args.model
print(modelName)
def updataParameters(load_dict,d_model):
    model_dict = d_model.state_dict()
    new_list = list(d_model.state_dict().keys()) 
    #model_dict[new_list[0]]
    pretrained_dict = {k: v for k, v in load_dict.items() if k in model_dict}#filter out unnecessary keys 
    model_dict.update(pretrained_dict)
    d_model.load_state_dict(model_dict)
model_feat = relationNet_multi_head_corr_row_hN(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
model_fc = relationNet_multi_head_corr_row_hN_fc(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
# Load the pretrained model
if args.cuda:
    if args.gpu_id !=-1:
        torch.cuda.set_device(args.gpu_id)
    model_feat = model_feat.cuda()
    model_feat = nn.DataParallel(model_feat)
    model_fc = model_fc.cuda()
    model_fc = nn.DataParallel(model_fc)
model_name = modelName
dict_train = torch.load(model_name)
model_feat.load_state_dict(dict_train)
# dict_new = model_fc.state_dict()
# new_list = list(model_fc.state_dict().keys())
# trained_list = list(dict_train.keys())
# copyLength=len(new_list)
# for k in range(copyLength):
#     dict_new[new_list[k]] = dict_train[trained_list[-(copyLength-k)]]
# model_fc.load_state_dict(dict_new)
updataParameters(dict_train,model_fc)
# Extracting features
def extract_feature():
    model_feat.eval()
    model_fc.eval()
    src_feat = []
    face_feat = []
    faceCon_feat = []
    coor_feat = []
    label_test = []
    ImgId_test = []
    for test_batch_idx, (test_src, test_face, test_faceCont, test_coor,  test_label, test_ImgId) in enumerate(test_loader):
        if args.cuda:
            test_src, test_face, test_faceCont, test_coor = test_src.cuda(), test_face.cuda(), test_faceCont.cuda(), test_coor.cuda()
        src_, face_ = model_feat(test_src, test_face, test_faceCont,test_coor)
        src_ = src_.data.cpu()
        face_ = face_.data.cpu()
        src_feat.append(src_)
        face_feat.append(face_)
        label_test.append(test_label)
        ImgId_test.extend(test_ImgId)
    src_feat = torch.cat(src_feat)
    face_feat = torch.cat(face_feat)
    test_label = torch.cat(label_test)
    src_feat_ = np.zeros((src_feat.size()))
    face_feat_ = np.zeros((face_feat.size()))
    test_label_ = np.zeros((src_feat.size(0),))
    ImgId_test_ = np.zeros((src_feat.size(0),))
    for count, (src_, face_, label_, ImgId_) in enumerate(
            zip(src_feat, face_feat, test_label,
                ImgId_test)):
        src_feat_[count, :] = src_.numpy()
        face_feat_[count, :] = face_.numpy()
        test_label_[count] = label_
        ImgId_test_[count] = ImgId_
    src_feat = src_feat_
    face_feat = face_feat_
    test_label = test_label_
    ImgId_test = ImgId_test_

    ImgId_dic = defaultdict(list)
    for index, (ImgId_) in enumerate(ImgId_test):
        ImgId_dic[ImgId_] = index
    uni_ImgId = list(ImgId_dic.keys())
    score = np.zeros((test_num,))
    f = open('../resultFile/result.txt', 'w')
    resultFile = open('../resultFile/temp.txt', 'w')
    prob=[]
    realLabel=[]
    flag=0
    imgID=[]
    for i in range(test_num):
        src_feat_ = src_feat[ImgId_test == uni_ImgId[i], :]
        face_feat_ = face_feat[ImgId_test == uni_ImgId[i], :]

        test_label_ = test_label[ImgId_test == uni_ImgId[i]]
  
        src_feat_ = Variable(torch.from_numpy(src_feat_).float().cuda())
        face_feat_ = Variable(torch.from_numpy(face_feat_).float().cuda())
        
        hat_label = model_fc(src_feat_, face_feat_ ,str(int(uni_ImgId[i])+1)+".jpg").data.cpu().numpy()
        num_face = len(test_label_)
        if len(hat_label[test_label_ == 1])==0:
            print("***********************no vip*********************")
            continue
        prob.append(hat_label)
        realLabel.append(np.argmax(test_label_)+1)
        imgID.append(int(uni_ImgId[i])+1)
        if np.argmax(hat_label) == np.argmax(test_label_) and len(hat_label[hat_label==hat_label[np.argmax(hat_label)]])==1:
            score[i] = 1
            f.write("Correct: " + str(uni_ImgId[i]) + '  response:' + str(np.argmax(hat_label, axis=0)) + '  probability:' + str(hat_label[test_label_ == 1][0]) + '  answer: ' + str(np.argmax(test_label_)) + '\n')
        else:
            print(str(i)+" th image response: "+str(np.argmax(hat_label, axis=0))+"    answer: "+str(np.argmax(test_label_)))
            f.write("Wrong: " + str(uni_ImgId[i]) + '  response:' + str(np.argmax(hat_label, axis=0)) + '  probability:' + str(hat_label[test_label_ == 1][0]) + '  answer: ' + str(np.argmax(test_label_)) + '\n')
        string=""
        for idx in range(len(hat_label)-1):
            string+=str(hat_label[idx])
            string+=","
        string+=str(hat_label[-1])
        resultFile.write(str(int(uni_ImgId[i])+1)+".jpg"+","+string+'\n')
    print('correct sum: '+str(np.sum(score)),'sum '+str(len(score)))
    print(np.mean(score))
    cmc, map = GetResults(copy.deepcopy(prob), realLabel)
    print(cmc, map)
    np.save('../resultFile/temp.npy',
            {'prob': prob,'isCorrect':score, 'label': realLabel,'imgID':imgID})
    resultFile.close()

start = time.time()
extract_feature()
end= time.time()
print('cost time:',(end-start)/test_num)
    # test()
