# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:42:08 2019

@author: Tanzeel
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
torch.manual_seed(42)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')

def train_One_epoch(model,train_loader,optimizer,loss_fn):
        # Sets model to TRAIN mode
        batch_losses,lrs = [],[]
        model.train()
        for p in model.parameters():
            p.requires_grad = True
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            #Makes predictions
            yhat = model(x_batch)
            # Computes loss
            loss = loss_fn(y_batch, yhat)
            # Computes gradients
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            #scheduler.step()  #Uncomment this for Cyclic and OneCycle LR
            batch_losses.append(loss.item())
            #lrs.append(scheduler.get_lr())
        training_loss = np.mean(batch_losses)
        # Returns the loss
        return training_loss,lrs
    
def evaluate(model,dataset_loader,loss_fn):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val, y_val in dataset_loader:
            yhat = model(x_val)
            val_loss = loss_fn(y_val, yhat).item()
            val_losses.append(val_loss)
        
        validation_loss = np.mean(val_losses)
        return validation_loss

def Get_perdiction(model,X,auto):
    model.eval()
    x=torch.from_numpy(X).float()
    predict_Y=model(x)
    predict_Y=predict_Y.detach().numpy()
    predict_Y=auto.inv_fit(predict_Y)
    return predict_Y
    
def Make_one_one_plot(model,X,Y_true,auto,labelling):
    #Invert the true values which were pre-processed
    Y_true=auto.inv_fit(Y_true)
    predicted_Y=Get_perdiction(model,X,auto)
    _= plt.scatter(Y_true, predicted_Y,label=labelling)
    #plt.plot([-2,3],[-2,3]) # Y = PredY line

#Some metrics
def huber(y_true, y_pred, delta=1.0):
	y_true = y_true.reshape(-1,1)
	y_pred = y_pred.reshape(-1,1)
	return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))

def benchmark(train_X,train_Y,val_X, val_Y,test_X,test_Y, model,auto):
    #Invert the true values which were pre-processed
    train_Y=auto.inv_fit(train_Y)
    val_Y=auto.inv_fit(val_Y)
    test_Y=auto.inv_fit(test_Y)
    
    #Get the perdictions
    train_predict_Y=Get_perdiction(model,train_X,auto)
    val_predict_Y=Get_perdiction(model,val_X,auto)
    test_predict_Y=Get_perdiction(model,test_X,auto)
    
    #Get the huber loss
    hub = huber(train_Y, train_predict_Y)
    hub_val = huber(val_Y,val_predict_Y)
    #Get R2
    lr = LinearRegression()
    #Fit Linear regression on Training Data
    lr.fit(train_Y,train_predict_Y)
    R2_train=lr.score(train_Y,train_predict_Y)
    #Fit Linear Regression on val data
    lr.fit(val_Y,val_predict_Y)
    R2_Val=lr.score(val_Y,val_predict_Y)
    #Fit Linear Regression on Test data
    lr.fit(test_Y,test_predict_Y)
    R2_test=lr.score(test_Y,test_predict_Y)
    #Get MAPE
    mape_train=np.mean(np.abs((train_Y - train_predict_Y)/train_Y))*100
    mape_val=np.mean(np.abs((val_Y - val_predict_Y)/val_Y))*100
    #Get RMSE
    rms_train=np.sqrt(mean_squared_error(train_Y,train_predict_Y))
    rms_val=np.sqrt(mean_squared_error(val_Y,val_predict_Y))
    rms_test=np.sqrt(mean_squared_error(test_Y,test_predict_Y))
    #Print results
    print ("R2  Train/Val/Test\t%0.4F\t%0.4f\t%0.4F"%(R2_train, R2_Val,R2_test))
    print ("RMSE  Train/Val/Test\t%0.4F\t%0.4f\t%0.4F"%(rms_train, rms_val,rms_test))
    print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_val))
    print ("MAPE Train/Test\t%0.4F\t%0.4F"%(mape_train, mape_val))

def dataaugment(x, betashift = 0.05, slopeshift = 0.05,multishift = 0.05):
    #Shift of baseline
    #calculate arrays
    np.random.seed(42)
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x        
