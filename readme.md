## 简介

本仓库原本是为了对FIGAD论文的实验而创建的。

后来就在这个基础上进行新的实验。

### 1 Dataset

#### Brain Tumor Classification (MRI)

数据集下载：https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet

代码参考：https://www.kaggle.com/code/atharvamuley25/mobilenet-brain-tumor-mri-classification


### 2 联邦域适应


adv_adaption.py是利用gan进行对抗学习，PPT在E:\Doctor1\Latex\FedAdvDA\PPt.pptx
![alt text](image-1.png)

adv_adaption_cgan.py是条件gan加入label信息

不过都是在单个节点运行的，至于联邦的版本，在G2G中