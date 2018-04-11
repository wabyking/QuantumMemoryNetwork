from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import cPickle
import time,os,random
import itertools

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss


from collections import OrderedDict
class QuantumMemoryNetwork(nn.Module):
    def __init__(self, opt ):
        super(QuantumMemoryNetwork, self).__init__()
     

        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        # if opt.__dict__.get("embeddings",None) is not None:
        #     print('load embedding')
        #     self.encoder.weight=nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
        
        
 
    
    def getInitRho(self):        
        return Variable( torch.eye(opt.embedding_dim,opt.embedding_dim).repeat(opt.batch_size,1,1))
    
    def forward(self,x):

        content=self.encoder(x)      # batch_size * seq_size * embedding_dim
        self.content = content.view((-1,content.shape[-1]))    #(batch_size * seq_size) * embedding_dim

        projectedRho = torch.bmm(self.content.unsqueeze(2), self.content.unsqueeze(1))  
        #(batch_size * seq_size) * embedding_dim* embedding_dim
        
        projectedRho = projectedRho.view(-1,x.shape[1],projectedRho.shape[-2],projectedRho.shape[-1])
         # batch_size * seq_size * embedding_dim* embedding_dim
        states = []
        self.rho = self.getInitRho()
        for i in range(x.shape[1]):
            p=0.1
            self.rho = self.rho * (1-p)  +  projectedRho[:,i,:,:]
            states.append(self.rho)
        return states
    
    def train(self,x,y):
        states = self.forward(x)
        states = torch.cat(states,0)

        y=y.view(-1)
        self.y_embed = self.encoder(y)
        
        loss = torch.sum(torch.bmm(states,self.y_embed.unsqueeze(2)).squeeze())

        return loss
   
    
    def predict(self,x):        
        return None
    
    def predictbyBeamSearch(self,x):
        return None
    
class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.allowDotting()
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
            
if __name__ == '__main__':

    opt=DottableDict()
    opt.batch_size=10
    opt.vocab_size=2501
    opt.label_size=3
    opt.embedding_dim=100
    print(opt)
    model=QuantumMemoryNetwork(opt)


    sent = torch.arange(0,2500).view(10,250)
    x= torch.autograd.Variable(sent[:,:-1].contiguous().long())
    y= torch.autograd.Variable(sent[:,1:].contiguous().long())
    loss= model.train( x, y)
    o = model(x)




