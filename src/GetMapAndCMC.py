
import numpy as np
import pdb

from collections import defaultdict
import scipy.io as scio

def GetResults(Scores, Label, rank=20, n_good=1):
    # Scores is a list, Scores[0] represent the scores of the first image

    # Compute CMC
    num_Image = len(Scores)
    if num_Image != len(Label):
        print('The Scores can not match the Input Label!')
    cmc = np.zeros((num_Image,rank))
    ap = np.zeros((num_Image,))
    for i in range(num_Image):
        old_recall = 0.0
        old_precision = 1.0
        good_now = 0.0
        intersect_size = 0.0
        score = Scores[i]
        index = Label[i]-1
        num_persons = len(score)
        for j in range(num_persons):
            flag = 0
            if score[index] == max(score) and len(score[score==score[index]])==1:
            # if np.max(score[index == 1]) == np.max(score):
                cmc[i,j:] = 1.0
                flag = 1
                good_now += 1
            else:
                score[score==max(score)] = -1
            if flag == 1:
                intersect_size += 1.0
            recall = intersect_size / n_good
            precision = intersect_size / (j+1)
            ap[i] = ap[i] + (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision
            if good_now == n_good:
                break
    CMC = np.mean(cmc,0)
    Map = np.mean(ap)
    return CMC, Map



def GetClassifyAcc(Scores, Label):
    # Scores is a list, Scores[0] represent the scores of the first image
    total_person=0
    correct_person=0
    correct_image=0
    # Compute CMC
    num_Image = len(Scores)
    # if num_Image != len(Label):
    #     print('The Scores can not match the Input Label!')
    for i in range(num_Image):
        score = Scores[i]
        index = Label[i]-1
        num_persons = len(score)
        image_false=0
        for j in range(num_persons):
            if index == j:
                if score[j]<=0.5:
                    image_false=1
                else:
                    correct_person+=1
            else:
                if score[j]<=0.5:
                    correct_person+=1
                else:
                    image_false=1
        if image_false==0:
            correct_image+=1
        total_person=total_person+num_persons
    pro1 = correct_image/num_Image
    pro2 = correct_person/total_person
    return pro1, pro2

# result =np.load('resultFile/new-permute-personRank-ProbTXT.npy').tolist()
# prob =result['prob']
# label = result['label']
# cmc,map = GetResults(prob,label)
# print(cmc,map)


def GetCatagoryResult(scores,Label,imgId):
    data = np.load('eventLabel.npy').tolist()
    index_test = data['index_test']
    eventLabel_test=data['eventLabel_test']
    print(len(index_test))
    event_dic = defaultdict(list)
    for i in range(len(index_test)):
        event_dic[index_test[i]]=eventLabel_test[i]
    if len(index_test)!=len(Label) or len(Label)!=len(imgId):
        print("error! The index_test can not match the Input Label!")
    for event in range(1,7):
        sub_scores,sub_label=[],[]
        for i in range(len(Label)):
            if event_dic[imgId[i]]==event:
                sub_scores.append(scores[i])
                sub_label.append(Label[i])
        cmc,map = GetResults(sub_scores,sub_label)
        print(cmc,map)
    sub_scores,sub_label=[],[]
    for i in range(len(Label)):
        if event_dic[imgId[i]]==8:
            sub_scores.append(scores[i])
            sub_label.append(Label[i])
    cmc,map = GetResults(sub_scores,sub_label)
    print(cmc,map)

def GetNCAACatagoryResult(scores,Label,imgId):
    DataName1 = '/home/sysu_issjyin_2/fating/NCAA/data/ImageFace_test.mat'
    data = scio.loadmat(DataName1)
    ImageFace = data['ImageFace_test']
    NumImage = ImageFace.shape[1]
    event_dic = defaultdict(list)
    image = []
    for i in range(NumImage):
        ImageName = int(ImageFace[0, i][0][0][:4])
        image.append(ImageName)
        event_label = ImageFace[0, i][2][0][0]['eventLabel'][0][0]
        event_dic[ImageName] = event_label
    if len(image)!=len(Label) or len(Label)!=len(imgId):
        print("error! The index_test can not match the Input Label!")
    for event in range(1,11):
        sub_scores,sub_label=[],[]
        for i in range(len(Label)):
            if event_dic[imgId[i]]==event:
                sub_scores.append(scores[i])
                sub_label.append(Label[i])
        cmc,map = GetResults(sub_scores,sub_label)
        print("===================="+'event '+str(event)+"=========================")
        print(cmc,map)
        print("=======================================================")
