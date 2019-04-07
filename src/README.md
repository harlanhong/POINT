#POINT
This repository contains an official pytorch implementation for the following paper
[Learning to Learn Relation for Important People Detection in Still Images (CVPR 2019)](http://TODO). Wei-Hong Li, Fa-Ting Hong, Wei-Shi Zheng
POINT, deep im**PO**rtance relat**I**on **N**e**T**work, is the first to investigate deep learning for exploring and encoding the relation features and exploiting them for important ppeople detection and achieves state-of-the-art performance on two public datasets for which verify its efficacy for important people detection
<!--TODO-->
##Citation:
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
<font color=#FF0000 >TODO</font>
##Dependencies
PyTorch  v1.0

##DataSet
In our work, we apply our algrithm to the [MS Dataset](http://TODO) and [NCAA Dataset](http://TODO), and achieve satisfactory results. 
##Prepare
As mentioned in our paper, the data feed into the network are some patches extracted from the original images, so that we have to do some preprocessing on the image. We use two different programs to process two different data sets, because the two data sets are detected in a different way, one is to detect the head and the other is to detect the whole body.
#####For [Ms Dataset](http://TODO)
python GetMSImageFace.py
#####For [NCAA Dataset](http://TODO)
python GetNCAAImageFace.py

##Training
python POINT_train.py 
