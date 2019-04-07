from torch import nn
import torch
import torchvision
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import pdb
import numpy as np
from PIL import Image
# from relationNetworkModule import relation_network
# import cv2
from torchvision import transforms




###############################################################################################################

##baseline inter+loca+exter
class single_fa_classification(nn.Module):
    def __init__(self,num_classes=0, num_instances=1, **kwargs):
        super(single_fa_classification, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
     
        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self,src,face,faceCon,Coor):
        Coor =Coor.unsqueeze(1)
        faceCon_ = self.faceConNet(faceCon)
        face_ = self.faceNet(face)
        coor=self.CoorConv(Coor)
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        if not self.training:
            return x,x
        prelogits = self.classifier(x)
        return prelogits
class single_fa_classification_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,**kwargs):
        super(single_fa_classification_fc, self).__init__()
        self.W_feature = []
        self.Img_Name = []
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.hidden=256
        self.feat_dim = 1024  
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        if not self.training:
            prelogits = self.classifier(face)
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            return self.hat_label

#baseline inter
class single_face_classification(nn.Module):
    def __init__(self,num_classes=0, num_instances=1, **kwargs):
        super(single_face_classification, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]
        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(2048,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
     
        self.relu = nn.ReLU()
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self,src,face,faceCon,Coor):
        face_ = self.faceNet(face)
        roifeat=self.ROIConv(face_)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        if not self.training:
            return x,x
        prelogits = self.classifier(x)
        return prelogits
class single_face_classification_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,**kwargs):
        super(single_face_classification_fc, self).__init__()
        self.W_feature = []
        self.Img_Name = []
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.hidden=256
        self.feat_dim = 1024  
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        if not self.training:
            prelogits = self.classifier(face)
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            return self.hat_label

#baseline inter+loca
class single_face_coor_classification(nn.Module):
    def __init__(self,num_classes=0, num_instances=1, **kwargs):
        super(single_face_coor_classification, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]
        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(2304,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
     
        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self,src,face,faceCon,Coor):
        Coor =Coor.unsqueeze(1)
        face_ = self.faceNet(face)
        coor=self.CoorConv(Coor)
        ROIConv=torch.cat((face_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        if not self.training:
            return x,x
        prelogits = self.classifier(x)
        return prelogits
class single_face_coor_classification_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,**kwargs):
        super(single_face_coor_classification_fc, self).__init__()
        self.W_feature = []
        self.Img_Name = []
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.hidden=256
        self.feat_dim = 1024  
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        if not self.training:
            prelogits = self.classifier(face)
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            return self.hat_label

#w_mn 做softmax 再******点乘*******以一个w_v*fa，然后cat起来加fa，然后直接fc ,bn,fc分类,拥有不同的h和N 
class relationNet_multi_head_corr_row_hN_without_global(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_without_global, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        # linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    def attention_module_multi_head(self, roi_feat,img_feat,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        # img_embedding = img_embedding.cuda()
        # img_feat_1 = linear_global(img_embedding)
        # w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        if self.relation == 0:  # sub
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), -q_data))
        elif self.relation == 1:  # add
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), q_data))
        elif self.relation == 2:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) * q_data)
        elif self.relation == 3:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) / (q_data + 1e-6))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        # w_a_=[]
        # for i in range(w_a.size(0)):
        #     w_a_.append(w_a[i]*w_global[i])
        # w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.faceConNet(faceCon)
        face_ = self.faceNet(face)
        coor=self.CoorConv(Coor)
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(face.size(0) / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_,self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat  
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_corr_row_hN_without_global_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_without_global_fc, self).__init__()
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        # linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            src,self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self, roi_feat, img_feat):
        # roi_feat=self.linear_P(roi_feat)
        # img_feat=self.linear_I(img_feat)
        n, d = roi_feat.size()
     
        w_img = torch.add(roi_feat, img_feat.expand(n, d))
        
        return w_img
    def attention_module_multi_head(self, roi_feat,img_feat,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        # #Eq 5
        # img_embedding = img_embedding.cuda()
        # img_feat_1 = linear_global(img_embedding)
        # w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        if self.relation == 0:  # sub
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), -q_data))
        elif self.relation == 1:  # add
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), q_data))
        elif self.relation == 2:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) * q_data)
        elif self.relation == 3:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) / (q_data + 1e-6))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        # #Eq 3 Eq 2
        # w_a_=[]
        # for i in range(w_a.size(0)):
        #     w_a_.append(w_a[i]*w_global[i])
        # w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.W_feature,self.Img_Name

#POINT scale dot product
class relationNet_multi_head_corr_row_hN_scale_dot(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_scale_dot, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        # self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            for j in range(q_data.size(0)):
                aff.append(torch.dot(k_data[i],q_data[j]))
        aff = torch.stack(aff)
        w_a = aff.view(n,n,1)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.faceConNet(faceCon)
        face_ = self.faceNet(face)
        coor=self.CoorConv(Coor)
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(face.size(0) / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat  
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_corr_row_hN_scale_dot_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_scale_dot_fc, self).__init__()
        self.W_feature = []
        self.Img_Name = []
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        # self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    
    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            for j in range(q_data.size(0)):
                aff.append(torch.dot(k_data[i],q_data[j]))
        aff = torch.stack(aff)
        w_a = aff.view(n,n,1)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.W_feature,self.Img_Name




class relationNet_multi_head_corr_row_global_hN(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_global_hN, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_g = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.linear_g=nn.ModuleList(linear_g)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v,linear_g):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column

        fr_global = linear_g(img_feat)
        fr_global = w_global*fr_global

        return fr_sum+fr_global

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.faceConNet(faceCon)
        del faceCon
        face_ = self.faceNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        coor=self.CoorConv(Coor)
        del Coor
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        del face_,faceCon_,coor
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],
                                self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j],
                                self.linear_g[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_corr_row_global_hN_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_global_hN_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_g = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.linear_g=nn.ModuleList(linear_g)

        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],
                            self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j],
                            self.linear_g[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
   
    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v,linear_g):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column

        fr_global = linear_g(img_feat)
        fr_global = w_global*fr_global

        return fr_sum+fr_global

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w


#POINT inter
class relationNet_multi_head_face_corr_row_hN(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_corr_row_hN, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(2048,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

 
        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        ROIConv = self.faceNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_face_corr_row_hN_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_corr_row_hN_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)

        w_global = self.relu(img_feat_1)   #[n,1]
        #################TODO#####################
        # self.store_w_global.append(w_global)
        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
     
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
    
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        # self.store_w_a.append(w_a)
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        # self.store_w.append(w_a)
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w
#POINT inter+loca+exter without global information
class relationNet_multi_head_face_corr_row_hN_without_global(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_corr_row_hN_without_global, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        # self.faceConNet = torchvision.models.resnet50(pretrained=True)
        # self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(2048,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.relu = nn.ReLU()
        # linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat,img_feat,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        # img_embedding = img_embedding.cuda()
        # img_feat_1 = linear_global(img_embedding)
        # w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        if self.relation == 0:  # sub
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), -q_data))
        elif self.relation == 1:  # add
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), q_data))
        elif self.relation == 2:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) * q_data)
        elif self.relation == 3:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) / (q_data + 1e-6))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        # w_a_=[]
        # for i in range(w_a.size(0)):
        #     w_a_.append(w_a[i]*w_global[i])
        # w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        # faceCon_ = self.faceConNet(faceCon)
        ROIConv = self.faceNet(face)
        # coor=self.CoorConv(Coor)
        # ROIConv=torch.cat((face_,faceCon_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(face.size(0) / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_,self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat  
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_face_corr_row_hN_without_global_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_corr_row_hN_without_global_fc, self).__init__()
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        # linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            src,self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    def attention_module_multi_head(self, roi_feat,img_feat,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        # #Eq 5
        # img_embedding = img_embedding.cuda()
        # img_feat_1 = linear_global(img_embedding)
        # w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        if self.relation == 0:  # sub
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), -q_data))
        elif self.relation == 1:  # add
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), q_data))
        elif self.relation == 2:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) * q_data)
        elif self.relation == 3:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) / (q_data + 1e-6))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        # #Eq 3 Eq 2
        # w_a_=[]
        # for i in range(w_a.size(0)):
        #     w_a_.append(w_a[i]*w_global[i])
        # w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.W_feature,self.Img_Name

#POINT inter+loca
class relationNet_multi_head_face_coor_corr_row_hN(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_coor_corr_row_hN, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(2304,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        face_ = self.faceNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        coor=self.CoorConv(Coor)
        del Coor
        ROIConv=torch.cat((face_,coor),1)
        del face_,coor
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_face_coor_corr_row_hN_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_coor_corr_row_hN_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)

        w_global = self.relu(img_feat_1)   #[n,1]
        #################TODO#####################
        # self.store_w_global.append(w_global)
        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
     
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
    
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        # self.store_w_a.append(w_a)
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        # self.store_w.append(w_a)
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w

# POINT inter+loca+exter
class relationNet_multi_head_corr_row_hN(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n,1]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,1]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor,sample_num=8):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.faceConNet(faceCon)
        del faceCon
        face_ = self.faceNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        coor=self.CoorConv(Coor)
        del Coor
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        del face_,faceCon_,coor
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size /sample_num)):
                roifeat_ = roifeat[i *sample_num:(i + 1) *sample_num]
                img_ = src_[i *sample_num:(i + 1) *sample_num]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_corr_row_hN_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)

        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)

        w_global = self.relu(img_feat_1)   #[n,1]
        #################TODO#####################
        # self.store_w_global.append(w_global)
        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
     
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
    
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        # self.store_w_a.append(w_a)
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        # self.store_w.append(w_a)
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w

#POINT inter+loca without global information
class relationNet_multi_head_face_coor_corr_row_hN_without_global(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_coor_corr_row_hN_without_global, self).__init__()
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        # self.faceConNet = torchvision.models.resnet50(pretrained=True)
        # self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(2304,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        # linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat,img_feat,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        # img_embedding = img_embedding.cuda()
        # img_feat_1 = linear_global(img_embedding)
        # w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        if self.relation == 0:  # sub
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), -q_data))
        elif self.relation == 1:  # add
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), q_data))
        elif self.relation == 2:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) * q_data)
        elif self.relation == 3:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) / (q_data + 1e-6))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        # w_a_=[]
        # for i in range(w_a.size(0)):
        #     w_a_.append(w_a[i]*w_global[i])
        # w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,128]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        # faceCon_ = self.faceConNet(faceCon)
        face_ = self.faceNet(face)
        coor=self.CoorConv(Coor)
        ROIConv=torch.cat((face_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(face.size(0) / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_,self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat  
        x = self.subspace(roifeat)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_face_coor_corr_row_hN_without_global_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_face_coor_corr_row_hN_without_global_fc, self).__init__()
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            src,self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        # roi_feat=self.linear_P(roi_feat)
        # img_feat=self.linear_I(img_feat)
        n,d = roi_feat.size()
        
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
       
        return w_img

    def attention_module_multi_head(self, roi_feat,img_feat,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        # #Eq 5
        # img_embedding = img_embedding.cuda()
        # img_feat_1 = linear_global(img_embedding)
        # w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        if self.relation == 0:  # sub
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), -q_data))
        elif self.relation == 1:  # add
            for i in range(k_data.size(0)):
                aff.append(torch.add(k_data[i].expand(n, d), q_data))
        elif self.relation == 2:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) * q_data)
        elif self.relation == 3:  # mul
            for i in range(k_data.size(0)):
                aff.append(k_data[i].expand(n, d) / (q_data + 1e-6))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        # #Eq 3 Eq 2
        # w_a_=[]
        # for i in range(w_a.size(0)):
        #     w_a_.append(w_a[i]*w_global[i])
        # w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.W_feature,self.Img_Name


##########################################################################################
# for rebuttal 
class relationNet_multi_head_corr_row_hN_rebuttal(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_rebuttal, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_qe=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_ke=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_qe=nn.ModuleList(linear_qe)
        self.linear_ke=nn.ModuleList(linear_ke)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
   
    def attention_module_multi_head(self, roi_feat,img_feat,linear_global,linear_q,linear_k,linear_qe,linear_ke,linear_a,linear_v):
  
        #Eq 5
        roi_feat_=linear_qe(roi_feat)
        img_feat_=linear_ke(img_feat)
        n, d = roi_feat_.size()
        img_embedding = torch.add(roi_feat_, img_feat_.expand(n, d))
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n,1]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,1]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.faceConNet(faceCon)
        del faceCon
        face_ = self.faceNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        coor=self.CoorConv(Coor)
        del Coor
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        del face_,faceCon_,coor
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_,self.linear_global[k*self.h+j],self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_qe[k*self.h+j],self.linear_ke[k*self.h+j],self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_corr_row_hN_rebuttal_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_rebuttal_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_qe=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_ke=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_qe=nn.ModuleList(linear_qe)
        self.linear_ke=nn.ModuleList(linear_ke)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            src,self.linear_global[i*self.h+j],self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_qe[i*self.h+j],self.linear_ke[i*self.h+j],self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def attention_module_multi_head(self, roi_feat,img_feat,linear_global,linear_q,linear_k,linear_qe,linear_ke,linear_a,linear_v):

        #Eq 5
        roi_feat_=linear_qe(roi_feat)
        img_feat_=linear_ke(img_feat)
        n, d = roi_feat_.size()
        img_embedding = torch.add(roi_feat_, img_feat_.expand(n, d))
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n,1]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,1]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w

class relationNet_multi_head_corr_row_hN_without_w_p2p(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_without_w_p2p, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.faceNet = torchvision.models.resnet50(pretrained=True)
        self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        self.faceConNet = torchvision.models.resnet50(pretrained=True)
        self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        # linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        # self.linear_q=nn.ModuleList(linear_q)
        # self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        # roi_feat=self.linear_P(roi_feat)
        # img_feat=self.linear_I(img_feat)
        n,d = roi_feat.size()
        
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
       
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = roi_feat  # [num_rois, 1024]
        k_data = roi_feat.clone()  # [num_rois, 1024]

        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n,1]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,1]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.faceConNet(faceCon)
        del faceCon
        face_ = self.faceNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        coor=self.CoorConv(Coor)
        del Coor
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        del face_,faceCon_,coor
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size / self.num_instances)):
                roifeat_ = roifeat[i * self.num_instances:(i + 1) * self.num_instances]
                img_ = src_[i * self.num_instances:(i + 1) * self.num_instances]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class relationNet_multi_head_corr_row_hN_without_w_p2p_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(relationNet_multi_head_corr_row_hN_without_w_p2p_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        # linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        # linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        # self.linear_q=nn.ModuleList(linear_q)
        # self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        params = list(self.parameters())
        k = 0
        for i in params:
            l = 1
            print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
                print("该层参数和：" + str(l))
            k = k + l
        print("总参数数量和：" + str(k))

        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        # roi_feat=self.linear_P(roi_feat)
        # img_feat=self.linear_I(img_feat)
        n,d = roi_feat.size()
        
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
       
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_a,linear_v):
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)

        w_global = self.relu(img_feat_1)   #[n,1]
        #################TODO#####################
        # self.store_w_global.append(w_global)
        # Eq 4
        q_data = roi_feat  # [num_rois, 1024/h]
        k_data = roi_feat.clone()  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
     
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
    
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        # self.store_w_a.append(w_a)
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        # self.store_w.append(w_a)
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w
