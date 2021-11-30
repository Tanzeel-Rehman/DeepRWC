# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:27:51 2019

@author: Tanzeel
"""

from train_Customdata import Load_Spectra
from torch.utils.data import DataLoader
from Inception_model import Inception
#from model import SampleCNN
from utils import init_weights
import torch.nn as nn
from LR_finder import LRFinder
import torch.optim as optim

#from Ranger_Optimizer import Lookahead_2 # Check on FastAI
import numpy as np

np.random.seed(42)

#filename = 'PLSmodeldata_ForTanzeel_101_removed_4 Classes_Full Data_34_wv_excluded.xlsx'
filename = '13_1+13_4_Top_June15.xlsx'
ls=Load_Spectra(filename)
weight_dir='../Deep learning approach/Weights/LR_Finder_Results/'

#Augment the data
train_X,train_Y,val_X,val_Y=ls.Augment(1,0.067)

'''noise_val = np.random.normal(0, 0.1, val_X.shape)
noise_train = np.random.normal(0, 0.1, train_X.shape)
train_X=train_X+noise_train
val_X=val_X+noise_val
'''
#train_X,train_Y,val_X,val_Y=ls.Get_train_val()
#Convert data to tensor
train_data = ls.XY_tensor(train_X,train_Y)
val_data = ls.XY_tensor(val_X,val_Y)

#Load the data
train_loader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=64,shuffle=True)


model = Inception()
model = model.apply(init_weights)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(),lr=1e-7)
#optimizer = optim.SGD(model.parameters(),lr=1e-7,momentum=0.9)

lr_finder = LRFinder(model, optimizer, criterion, device="cpu",memory_cache=False, cache_dir=weight_dir)
lr_finder.range_test(train_loader,val_loader,end_lr=100, num_iter=100,step_mode="exp",smooth_f=0.05,diverge_th=5)
lr_finder.plot()
