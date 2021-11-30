# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:10:16 2019

@author: Tanzeel
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN,BorderlineSMOTE,SVMSMOTE,KMeansSMOTE
from imblearn.combine import SMOTEENN,SMOTETomek

np.random.seed(42)

class Data_balance():
    def __init__(self,filename):
        #Matfile name
        self.filename=filename  
        
    def Get_XY(self):
        '''
        Return the spectral signature and groundtruth data from the given excel file.
        Make sure the columns of file containing spectral data have numeric labels such as wavelength, wavenumber
        he groundtruth can to be alphabetic, alphanumeric or numeric with or without the special characteristics.
        '''
        df=pd.read_excel(self.filename,engine='openpyxl')
        #Make a copy of data frame
        df2=df.copy()
        #Keep only those column headers which are numeric, turn all others to nan
        df2.columns = pd.to_numeric(df.columns, errors = 'coerce')
        #Remove all the 
        df2=df2[df2.columns.dropna()]
        
        X=df2
        
        Y=df['RWC']
        
        return X, Y

    def upsampled (self):
        '''
        This function is used to randomly sample the values of the under represented class (e.g. RWC < 80)
        In other words, it repeats the randomly picked variable to increase the data size. 
        '''
        X,X2 = self.Get_XY() # X is spectra and X2 is RWC
        
        df=pd.read_excel(self.filename,engine='openpyxl')
        #Combine the Spectrum and response variable as 1 df
        X=pd.concat([X,X2],axis=1)
        y = df.Class
        #Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        # concatenate our training data back together
        X2 = pd.concat([X_train, y_train], axis=1)
        # separate minority and majority classes
        low_RWC=X2[X2.Class==0]
        high_RWC=X2[X2.Class==1]
        # upsample minority
        RWC_upsampled = resample(low_RWC,
                                 replace=True, # sample with replacement
                                 n_samples=len(high_RWC), # match number in majority class
                                 random_state=42) # reproducible results
        
        # combine majority and upsampled minority
        upsampled = pd.concat([high_RWC, RWC_upsampled])
        #Seprate the spectrum and RWC data from the upsampled data
        X_train = upsampled.iloc[:,0:408].to_numpy()
        y_train = upsampled['RWC'].to_numpy()
        y_train=y_train.reshape((y_train.size,1))
        #Seprate the spectrum and RWC data from the Validation set
        X_val_1 = X_val.iloc[:,0:408].to_numpy()
        y_val = X_val['RWC'].to_numpy()
        #Change the shape y_val to be n x 1 vector
        y_val=y_val.reshape((y_val.size,1))        
        
        return X_train,y_train,X_val_1,y_val
    
    def Data_Syntehsis (self, method):
        '''
        This function is used to Synthesize the data for underrepresented classes (e.g. RWC < 80)
        '''
        X,X2 = self.Get_XY() # X is spectra and X2 is RWC
        
        df=pd.read_excel(self.filename,engine='openpyxl')
        #Combine the Spectrum and response variable as 1 df
        X=pd.concat([X,X2],axis=1)
        y = df.Class
        #Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if method == 'SMOTE':
            #Synthesis the minority class using SMOTE
            sm = SMOTE(sampling_strategy='not majority',random_state=42)
            X_train, y_train = sm.fit_resample(X_train,y_train)
        elif method == 'ADASYN':
             #Synthesis the minority class using SMOTE
             adasyn = ADASYN(random_state=42, ratio=1.0)
             X_train, y_train = adasyn.fit_resample(X_train,y_train)
        elif method== 'BorderlineSMOTE':
             bsmote=BorderlineSMOTE(sampling_strategy='not majority',random_state=42)
             X_train, y_train = bsmote.fit_resample(X_train,y_train)
        elif method == 'KMeansSMOTE':
             ksmote=KMeansSMOTE(sampling_strategy='not majority',random_state=42)
             X_train, y_train = ksmote.fit_resample(X_train,y_train)
        elif method == 'SVMSMOTE':
             SVMsmote=SVMSMOTE(sampling_strategy='not majority',random_state=42)
             X_train, y_train = SVMsmote.fit_resample(X_train,y_train)
        elif method == 'SMOTETomek':
             smotetomek=SMOTETomek(sampling_strategy='not majority',random_state=42)
             X_train, y_train = smotetomek.fit_resample(X_train,y_train)
        elif method == 'SMOTEENN':
             smoteenn=SMOTEENN(sampling_strategy='not majority',random_state=42)
             X_train, y_train = smoteenn.fit_resample(X_train,y_train)
             X_val, y_val = smoteenn.fit_resample(X_val,y_val)
        else:
             print("No other method has been implemented")
        
        
        #Seprate the spectrum and RWC data from the synthesized data
        X_train_1 = X_train.iloc[:,0:408].to_numpy()
        y_train=X_train.iloc[:,408].to_numpy()
        #Chnage the shape of y_train to be a n x 1
        y_train=y_train.reshape((y_train.size,1)) 
        
        #Seprate the spectrum and RWC data from the Validation set
        X_val_1 = X_val.iloc[:,0:408].to_numpy()
        y_val = X_val.iloc[:,408].to_numpy()
        #Change the shape y_val to be n x 1 vector
        y_val=y_val.reshape((y_val.size,1))        
        
        return X_train_1,y_train,X_val_1,y_val


    ''' Obselete Function TR: Oct_10_2019'''

    '''
    def Syntehsis_ADASYN (self):
        df=pd.read_excel(self.filename)
        #Get the Spectrum
        X=df.iloc[:,45:558]
        #Get the RWC or any response variable
        X2=df['RWC']
        #Combine the Spectrum and response variable as 1 df
        X=pd.concat([X,X2],axis=1)
        y = df.Class
        #Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        #Synthesis the minority class using SMOTE
        adasyn = ADASYN(random_state=42, ratio=1.0)
        X_train, y_train = adasyn.fit_sample(X_train,y_train)
        #Seprate the spectrum and RWC data from the synthesized data
        X_train_1 = X_train[:,0:513]
        y_train=X_train[:,513]
        #Chnage the shape of y_train to be a n x 1
        y_train=y_train.reshape((y_train.size,1)) 
        
        #Seprate the spectrum and RWC data from the Validation set
        X_val_1 = X_val.iloc[:,0:513].to_numpy()
        y_val = X_val['RWC'].to_numpy()
        #Change the shape y_val to be n x 1 vector
        y_val=y_val.reshape((y_val.size,1))        
        
        return X_train_1,y_train,X_val_1,y_val
        '''