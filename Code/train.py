# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 02:53:21 2019

@author: Tanzeel
"""
#from DataLoader import Load_Spectra
from train_Customdata import Load_Spectra
import torch
import torch.nn as nn
import torch.optim as optim
from quantileloss import XTanhLoss  # A bunch of losses for Regression----Obselete TR: Nov_30_2019
from torch.utils.data import DataLoader
from Inception_model import Inception
import numpy as np
import matplotlib.pyplot as plt
from visualization_utils import Activate,Get_Kernels
from utils import init_weights,Make_one_one_plot,benchmark,train_One_epoch,evaluate

''' ---- Download these from FASTAI-----'''
#from Ranger_Optimizer import Lookahead_2
#from OneCycle_Policy import OneCycleLR    #Obselete TR: Dec_20_2020  (Pytorch has cyclic schedulers now) 

'''---------Below is Main code'''
torch.manual_seed(42)
np.random.seed(42)

# Define parameters for training and lists for holding losses
n_epochs = 2
training_losses = []
training_lrs=[]
validation_losses = []
best_valid=float("inf")


#File containing spectral data and RWC response vector
filename = '13_1+13_4_Top_June15.xlsx'
ls=Load_Spectra(filename)

#Get the Augmented Data 
train_X,train_Y,val_X,val_Y=ls.Augment(10,0.067)

#Make the data tensor
train_data = ls.XY_tensor(train_X,train_Y)
val_data = ls.XY_tensor(val_X,val_Y)

#Load the data using dataloader
train_loader=DataLoader(dataset=train_data,batch_size=512,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=512,shuffle=True)


#Load the Inception model 
model = Inception()
# He Random Weight initialization
model = model.apply(init_weights)
# MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

'''----- Test different optimizers with different scheduling options-----'''

optimizer = optim.Adam(model.parameters(),lr=0.00316)   # Baseline with fixed learning rate found from LR finder
#optimizer = optim.SGD(model.parameters(),lr=3.25643e-5,momentum=0.9)

'''--------Schduler For Cyclic LR-----------'''
"""
#1) Cyclic LR with cosine annelaing 
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0002633, max_lr=0.00316,step_size_up=3664, 
                                              step_size_down=None, mode='exp_range',gamma=0.99994,scale_fn=None,
                                              cycle_momentum=False)

#2) Cyclic LR with Triangular  
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0002633, max_lr=0.00316,step_size_up=3664, 
                                              step_size_down=None, mode='triangular',gamma=1.0,
                                              cycle_momentum=False)

"""

"""
1) Schduler For One Cycle Policy with Cosine Annealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.00316,total_steps=(n_epochs*457)+1,
                                                anneal_strategy='cos',pct_start=0.4,cycle_momentum=False,div_factor=12)

2) Scheduler For  One Cycle policy with Linear Annealing
scheduler = OneCycleLR(optimizer, num_steps=(457*n_epochs), lr_range=(0.00316/12, 0.00316),annihilation_frac = 0.2)

3) Schduler For One Cycle with SGD Optimizer and Cos annealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=3.25643e-5,total_steps=(n_epochs*258)+1,
                                                anneal_strategy='cos',pct_start=0.4,cycle_momentum=True,div_factor=5.0)

"""

'''------------Begine training the network-----------'''
for epoch in range(n_epochs):
    training_loss,lrs=train_One_epoch(model,train_loader,optimizer,loss_fn)  
    training_losses.append(training_loss)
    training_lrs.append(lrs)
    #Evaluate the model
    validation_loss=evaluate(model,val_loader,loss_fn)
    validation_losses.append(validation_loss)

    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
    
    #Check the losses and save the model
    if validation_loss < best_valid:
        best_valid = validation_loss
        print("New Best model found")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':training_loss,
                'val_loss':validation_loss
                },f"../Deep learning approach/Weights/test_wts.pt")
    
'''------------End of training loop---------------'''

#Display the training and validation losses
plt.figure()
plt.plot(training_losses,label='Training')
plt.plot(validation_losses,label='Validation')
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Train and Validation Loss',fontsize=16)
plt.legend(loc='best',fontsize = 16)

# The AutoScale generator for Y variable is called here to back scale the pre-processed Y data
auto = ls.Return_Y_yscaler()

# Display the measured and predicted RWC for both training and test datasets
plt.figure()
Make_one_one_plot(model,train_X,train_Y,auto,'Training')
Make_one_one_plot(model,val_X,val_Y,auto,'Val')
plt.xlabel('Measured RWC (%)', fontsize = 16)
plt.ylabel('Predicted RWC (%)', fontsize = 16)

'''----------Feature Visulaization------------'''
#Get the features of different convloutional layers and their mean activations
mean_act,features=Activate(model,0,train_X) # Here 0 represents the first convolutional layer
plt.figure()
_=plt.plot(mean_act,linewidth=2.)
plt.xlabel('Feature number', fontsize = 16)
plt.ylabel('Mean Activations', fontsize = 16)

#Get the first few highly active features only
b=np.array(mean_act)
b=b.argsort()[-5:][::-1]
b=features[b]
#Plot these features to see how they look like
plt.figure()
_=plt.plot(b.T)
plt.xlabel('Spectral Band Number', fontsize = 16)
plt.ylabel('Highly Active Feature Profile', fontsize = 16)

''' Visualize the kernels'''
#Get the weights of indvidual kernel and plot it
kernels=Get_Kernels(model,0,0)

'''-----------Test the model on external datasets---------------'''
#Load the best saved model and apply weights 
checkpoint=torch.load(f"../Deep learning approach/Weights/test_wts.pt")
model.load_state_dict(checkpoint['model_state_dict'])
#Load the test datasets
test_oct_28='13-4_Imaging_6ms_Oct_28_results.xlsx'
X_test_4,Y_test_4=ls.Get_Test_data(test_oct_28)
# Benchmark the test data
benchmark(train_X,train_Y,val_X,val_Y,X_test_4,Y_test_4,model,auto)
