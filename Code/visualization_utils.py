# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:29:07 2019

@author: Tanzeel
"""
import torch
import numpy as np
from matplotlib import pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.labelsize']=20.0
plt.rcParams['ytick.labelsize']=20.0


#Register the hooks for visualizing the model output
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        #self.features = torch.tensor(output,requires_grad=True)
        self.features = output.clone().detach()
    def close(self):
        self.hook.remove()


def plot_features(features,num_cols=6):
    num_features=features.shape[0]
    num_rows = 1+ num_features // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(features.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.plot(features[i].T)
        #ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.xlabel('Wavelength (nm)', fontsize = 20)
    plt.show()
    

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==3:
        raise Exception("assumes a 3D tensor")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.plot(tensor[i].T)
        #ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def Activate(model,layer,train_X):
    # Register the hooks on specific conv layer
    activations = SaveFeatures(list(model.children())[layer])
    # Turn on evaluation mode
    model.eval()
    #Compute the average spectra from the data and make it tensor. Pass through the network to store hook's data
    model(torch.from_numpy(np.mean(train_X,0)).float().view(1,-1))
    # Comute the mean activations for every feature. This will help to find features with highest contribution
    mean_act = [activations.features[0,i].mean().item() for i in range(activations.features.size(1))]
    # Tensor to the numpy
    feat=activations.features.data.numpy().squeeze()
    activations.close() # Close hooks
    plot_features(feat,8)
    return mean_act,feat

def Get_Kernels(model,layer,sub_layer):
    #kernels=model.layer[sub_layer].weight.detach().numpy().squeeze()
    #filters=model.modules
    model.eval()
    body_model = [i for i in model.children()][layer]
    layer1 = body_model[sub_layer]
    kernels=layer1.weight.data.numpy()
    plot_kernels(kernels,8)
    return kernels
