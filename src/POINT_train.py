from __future__ import print_function

import argparse
# from vps import VPS
import pdb
from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms as T

from dataset import *
from preprocessor import *
from relationNetwork import *
from sampler import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=700, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset-root', type=str, default='./',
                    help='directory of all images')
parser.add_argument('--nw', type=int, default=12,
                    help='number of workers')
parser.add_argument('--ni', type=int, default=8,
                    help='number of instances for each tracklet (default is 2)')
parser.add_argument('--tl', type=int, default=2,
                    help='number of tracklets for each identity (default is 2)')
parser.add_argument('--gpu-id', type=int, default=-1,
                    help='gpu id (default is 2)')
parser.add_argument('--log-dir', type=str, default='./',
                    help='directory of the log file (default is ./)')
parser.add_argument('--decay-epoch', type=int, default=100,
                    help='number of epoch per decay')
parser.add_argument('--h',type=int, default=4,help='the number of relation submodule')
parser.add_argument('--N',type=int,default=2,help='the number of relation module')
parser.add_argument('--save_iter',type=int,default=100,help='Save the model every save_iter epoch')
parser.add_argument('--save_name',type=str,default='POINT',help='the name of saved model')
parser.add_argument('--index_name',type=str,default='../data/MSindex.npy',help='the name of index of train, val and test set')
parser.add_argument('--dataset_path',type=str,default='../data/MSDataSet_process',help='the path of dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

log_dir = args.log_dir
log_file = open(log_dir+'log.txt','w')
log_file.write('**************************************\n')
log_file.write('Training on VPS using SVM with bn feat\n')
log_file.write('**************************************\n')

VIPDataset = RelatinNetCMSIP(root=args.dataset_path,index=args.index_name)

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
train_transformer1 = T.Compose([
        T.RandomResizedCrop((224), (0.9, 1)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
        ])

train_transformer2 = T.Compose([
        # T.ToPILImage(),
        T.RandomResizedCrop((224), (0.8, 0.85)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
        ])

train_transformer3 = T.Compose([
        # T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        normalizer
        ])

val_transformer1 = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                normalizer
                                ])

val_transformer2 = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                normalizer
                                ])

val_transformer3 = T.Compose([T.Resize((224, 224)),
                                T.ToTensor(),
                                normalizer
                                ])
#load train_loader and val_loader
train_set = VIPDataset.train  
val_set = VIPDataset.test       
train_num = VIPDataset.num_train  
val_num = VIPDataset.num_test

train_loader = DataLoader(
    RelationNetCPreprocessor(train_set, isTrain=True, 
                     transform1=train_transformer1,transform2=train_transformer2,transform3=train_transformer3),
                     sampler = RelationNetRandomFaceSampler(data_source=train_set,
                                                     num_instances=args.ni),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)

val_loader = DataLoader(
    RelationNetCPreprocessor(val_set, isTrain=False,
                     transform1=val_transformer1, transform2=val_transformer1,transform3=val_transformer3),
                     sampler = RelationNetTestSampler(data_source=val_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)

save_inter=20

#The path and name saved by the model
saveDir='./models/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
model_name=args.save_name+'.pkl'

print(model_name)
#load model
model = relationNet_multi_head_corr_row_hN(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
model_fc = relationNet_multi_head_corr_row_hN_fc(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)

if args.cuda:
    if args.gpu_id !=-1:
        torch.cuda.set_device(args.gpu_id)
    model = model.cuda()
    model = nn.DataParallel(model)
    model_fc = model_fc.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# weights = [1/7,1]
# class_weights = torch.FloatTensor(weights).cuda()
lossF = nn.CrossEntropyLoss().cuda()

#Decay the learning rate
def adjust_learning_rate(opt, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr * (0.5 ** (epoch // args.decay_epoch))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

#Traing
def train(epoch, val_acc,bestResult):
    accuracy=0
    num=0
    for batch_idx, (src,face, faceCont,coor,target, ImgId) in enumerate(train_loader):
        model.train()
        if args.cuda:
            src,face,faceCont,coor, target = src.cuda(),face.cuda(),faceCont.cuda(),coor.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(src,face,faceCont,coor)
        loss = lossF(output,target)
        #caculate the train accuracy
        prediction = torch.max(F.softmax(output), 1)[1]
        pred_y = prediction.data.cpu().numpy().squeeze()
        target_y = target.data.cpu().numpy()
        accuracy = sum(pred_y == target_y)+accuracy
        num=target.size(0) +num
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tVal Acc: {:.4f}\tTrain Acc: {:.2f}'.format(
                epoch, batch_idx * args.batch_size, train_num * args.ni,
                100. * batch_idx / len(train_loader),loss.item(), val_acc,train_acc))
            log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tVal Acc: {:.4f}\tTrain Acc: {:.2f}  \n'.format(
                epoch, batch_idx * args.batch_size, train_num * args.ni,
                100. * batch_idx / len(train_loader), loss.item(), val_acc,train_acc))
    if epoch > 0 and epoch % (4 * args.log_interval) == 0:
        save_name =saveDir+'/VIP_ResNet_%d_'%(epoch)+model_name
        if epoch%save_inter==0:
            torch.save(model.state_dict(),save_name)
        #Assign test model parameters
        model.eval()
        model_fc.eval()
        dict_train=model.state_dict()
        dict_new = model_fc.state_dict().copy()
        new_list = list(model_fc.state_dict().keys())
        trained_list = list(dict_train.keys())
        copyLength=len(new_list)
        for k in range(copyLength):
            dict_new[new_list[k]] = dict_train[trained_list[-(copyLength-k)]]
        model_fc.load_state_dict(dict_new)
        src_feat = []
        face_feat=[]
        val_label_eval = []
        ImgId_eval = []
        for val_batch_idx, (val_src,val_face, val_faceCont,val_coor, val_label, val_ImgId) in enumerate(val_loader):
            if args.cuda:
                val_src,val_face,val_faceCont, val_coor= val_src.cuda(),val_face.cuda(),val_faceCont.cuda(),val_coor.cuda()
            val_src, val_face,val_faceCont,  val_coor =Variable(val_src), Variable(val_face),Variable(val_faceCont),Variable(val_coor)
            src_, face_ = model(val_src,val_face,val_faceCont,val_coor)
            src_ = src_.data.cpu()
            face_ = face_.data.cpu()

            src_feat.append(src_)
            face_feat.append(face_)

            val_label_eval.append(val_label)
            ImgId_eval.extend(val_ImgId)
        #Tensor
        src_feat = torch.cat(src_feat)
        face_feat = torch.cat(face_feat)
        val_label_eval = torch.cat(val_label_eval)

        src_feat_ = np.zeros((src_feat.size()))
        face_feat_ = np.zeros((face_feat.size()))
        val_label_eval_ = np.zeros((src_feat.size(0),))
        ImgId_eval_ = np.zeros((src_feat.size(0),))
        for count, (src_,face_, label_, ImgId_) in enumerate(zip(src_feat,face_feat,val_label_eval, ImgId_eval)):
            src_feat_[count, :] = src_.numpy()
            face_feat_[count, :] = face_.numpy()
            val_label_eval_[count] = label_
            ImgId_eval_[count] = ImgId_
        src_feat = src_feat_
        face_feat = face_feat_

        val_label_eval = val_label_eval_
        ImgId_eval = ImgId_eval_

        ImgId_dic = defaultdict(list)
        for index, (ImgId_) in enumerate(ImgId_eval):
            ImgId_dic[ImgId_] = index
        uni_ImgId = list(ImgId_dic.keys())
        # pdb.set_trace()
        score = np.zeros((val_num,))
        for i in range(val_num):

            src_feat_ = src_feat[ImgId_eval == uni_ImgId[i], :]
            face_feat_ = face_feat[ImgId_eval == uni_ImgId[i], :]

            val_label_eval_ = val_label_eval[ImgId_eval == uni_ImgId[i]]
            src_feat_=Variable(torch.from_numpy(src_feat_).float().cuda())
            face_feat_=Variable(torch.from_numpy(face_feat_).float().cuda())

            hat_label = model_fc(src_feat_,face_feat_)
            hat_label=hat_label.data.cpu().numpy()
            num_face = len(val_label_eval_)
            if np.max(hat_label[val_label_eval_ == 1]) == np.max(hat_label) and len(hat_label[hat_label==hat_label[np.argmax(hat_label)]])==1:
                score[i] = 1
            # print('val: ',i, '; result: ',score[i])
        val_acc = np.mean(score)
        return val_acc,accuracy/(num+(1e-6)),bestResult

    return val_acc,accuracy/(num+(1e-6)),bestResult

    
val_acc = 0.0
val_acc_2=0.0
train_acc = 0.0
bestResult=0
for epoch in range(0, args.epochs):
    val_acc ,train_acc,bestResult= train(epoch, val_acc,bestResult)
    adjust_learning_rate(optimizer, epoch)
    print("=================optimizer lr==================")
    print(model_name)
log_file.close()
    # test()
