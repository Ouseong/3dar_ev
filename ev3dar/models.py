"""Penguin-style circular SSL, new spectrogram ID; no pretrained/attention code."""
import torch
from torch import nn
import torch.nn.functional as F


class LocalizationCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(channels,128,7,padding=3,padding_mode='circular'),nn.InstanceNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,256,5,padding=2,padding_mode='circular'),nn.InstanceNorm1d(256),nn.ReLU(),
            nn.Conv1d(256,256,5,padding=4,dilation=2,padding_mode='circular'),nn.InstanceNorm1d(256),nn.ReLU(),
            nn.Conv1d(256,128,3,padding=1,padding_mode='circular'),nn.InstanceNorm1d(128),nn.ReLU())
        self.count_branch=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,64),
                                       nn.ReLU(),nn.Dropout(.3),nn.Linear(64,3))
        self.loc_branch=nn.Sequential(nn.Conv1d(128,64,3,padding=1,padding_mode='circular'),
                                      nn.ReLU(),nn.Conv1d(64,1,1))

    def forward(self,x):
        feat=self.features(x)
        return self.count_branch(feat),self.loc_branch(feat).squeeze(1)


class SoundTypeCNN(nn.Module):
    def __init__(self,classes=6,use_rpm=False):
        super().__init__();self.use_rpm=use_rpm
        layers=[];previous=1
        for c in [32,64,128]:
            layers.extend([nn.Conv2d(previous,c,3,padding=1),nn.GroupNorm(8,c),nn.GELU(),nn.MaxPool2d(2)])
            previous=c
        self.encoder=nn.Sequential(*layers)
        self.fcn=nn.Sequential(nn.Linear(128+(2 if use_rpm else 0),128),nn.GELU(),nn.Dropout(.2),
                               nn.Linear(128,64),nn.GELU())
        self.decoder=nn.Linear(64,classes)  # Classification decoder, NOT waveform reconstruction.

    def forward(self,views,masks,durations,rpm):
        b,p,v,m,t=views.shape
        x=self.encoder(views.reshape(b*p*v,1,m,t))
        valid=F.adaptive_avg_pool1d(masks.reshape(b*p*v,1,t),x.shape[-1]).unsqueeze(2)
        z=(x*valid).sum((2,3))/(valid.sum((2,3))*x.shape[2]).clamp_min(1e-6)
        z=z.reshape(b,p,v,-1)
        weights=durations*(masks.sum(-1)>0).to(durations.dtype)
        z=(z*weights.unsqueeze(-1)).sum(2)/weights.sum(2,keepdim=True).clamp_min(1e-6)
        if self.use_rpm:z=torch.cat([z,rpm],dim=-1)
        latent=self.fcn(z)
        return self.decoder(latent),latent


def focal_loss(logits,target):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    return (.25*(1-torch.exp(-bce)).pow(2)*bce).mean()
