from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal




def PCELoss(y_1, t, forget_rate,device,criterion):
    outputs = F.softmax(y_1, dim=1)
    #index = np.nonzero(weight)
    #loss_1_update = criterion(y_1[index], t[index])
    #Lu = (F.cross_entropy(y_1, t,reduction='none') * weight).mean()
    
    loss_1 = F.cross_entropy(y_1, t,reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).to(device)
    loss_1_sorted = loss_1[ind_1_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    ind_1_update=ind_1_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]
    loss_1_update = criterion(outputs[ind_1_update], t[ind_1_update])
    return torch.sum(loss_1_update)