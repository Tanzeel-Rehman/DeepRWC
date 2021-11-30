# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 21:37:19 2019

@author: Tanzeel
"""

import torch
from torch.utils.data import TensorDataset
import numpy as np
from utils import dataaugment
from Pre_Process import Autoscale
from Data_balance import Data_balance

np.random.seed(42)

class Load_Spectra(TensorDataset):
    def __init__(self,filename):
        #File name
        self.filename=filename
        #Scale the X data 
        self.xscaler = Autoscale()
        #Scale the y data
        self.yscaler = Autoscale()
        

    def Get_train_val(self,method='SMOTEENN'):
        '''
        This function returns the Spectral data and response variable by removing the imbalance in the data
        No Augmentation is performed here
        '''
        db=Data_balance(self.filename)
        #Get the Balanced Spectra
        X_train, Y_train, X_val,Y_val=db.Data_Syntehsis(method)
        
        #Normalize the X training data
        X_train = self.xscaler.Calibrate(X_train)
        #Normalize the X Val data-- Apply training data statisitcs on val
        X_val = self.xscaler.Apply(X_val)
        
        #Normalize the Y training data
        Y_train = self.yscaler.Calibrate(Y_train)
        #Normalize the Y Val data-- Apply training data statisitcs on val
        Y_val = self.yscaler.Apply(Y_val)
        
        return X_train,Y_train,X_val,Y_val
    
    
    def Augment(self,reps,shift):
        '''
        This function is used to Augment the spectral data and the response variable
        '''
        
        # Get the normalized balanced data
        X_train,Y_train,X_val,Y_val = self.Get_train_val('SMOTEENN')
        
        #Repeats the training and validation spectrum 
        X_train_aug = np.repeat(X_train, repeats=reps, axis=0)
        X_val_aug = np.repeat(X_val, repeats=reps, axis=0)
        #Now modify the repeated data
        X_train_aug = dataaugment(X_train_aug, betashift = shift, slopeshift = (shift/2), multishift = shift)
        X_val_aug = dataaugment(X_val_aug, betashift = shift, slopeshift = (shift/2), multishift = shift)
        
        y_train_aug = np.repeat(Y_train, repeats=reps, axis=0) #y_train is simply repeated
        y_val_aug = np.repeat(Y_val, repeats=reps, axis=0) #y_train is simply repeated
        
        return X_train_aug,y_train_aug,X_val_aug,y_val_aug
    
    
    def Get_Test_data (self,testfilename):
        db = Data_balance(testfilename)
        X_test,Y_test=db.Get_XY()
        #Convert DF to arrays
        X_test = X_test.to_numpy()
        
        Y_test = Y_test.to_numpy()
        Y_test=Y_test.reshape((Y_test.size,1))
        
        #Normalize the X Val data-- Apply training data statisitcs on val
        X_test = self.xscaler.Apply(X_test)
        #Normalize the Y Val data-- Apply training data statisitcs on val
        Y_test = self.yscaler.Apply(Y_test)
        
        return X_test, Y_test
    
    def XY_tensor(self,X,Y):
        x_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(Y).float().view(-1,1)
        tensor_data=TensorDataset(x_tensor,y_tensor)
        return tensor_data
    
    def Return_Y_yscaler(self): #Used to scale back the predictions
        return self.yscaler