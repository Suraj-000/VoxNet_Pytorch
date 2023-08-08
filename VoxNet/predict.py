import os 
import torch
from voxnet import VoxNet
import numpy as np
from torch.utils.data import DataLoader

model_name="trained_weights/weights_15.pth"

CLASSES = {
        0: 'bathtub',
        1: 'chair',
        2: 'dresser',
        3: 'night_stand',
        4: 'sofa',
        5: 'toilet',
        6: 'bed',
        7: 'desk',
        8: 'monitor',
        9: 'table'
    }

voxnet = VoxNet(n_classes=len(CLASSES)) 
voxnet.load_state_dict(torch.load(model_name))

k="../ModelNet10_voxel/bed/train/bed_000000345.npy"
data=np.load(k)
data=data[np.newaxis,:]
data = data[np.newaxis, :]

l=DataLoader(data)

for j, sample in enumerate(l):
    voxel=sample
    if torch.cuda.is_available():
        voxel = voxel.cuda()
    voxel = voxel.float()  
    voxnet = voxnet.eval()
    pred = voxnet(voxel)

predc=pred.data.max(1)[1]
predc=predc.numpy()
print(CLASSES[predc[0]])
