import os 
import numpy as np
from torch.utils.data import Dataset
import glob

class ModelNet10VoxelDataset(Dataset):

    def __init__(self,root,classes,split="train"):
        self.root=root
        self.paths=[]
        self.classes2index={}
        for idx,cls in classes.items():
            self.classes2index.update({cls:idx})
            for path in glob.glob(os.path.join(self.root,cls,split,"*.npy")):
                self.paths.append(path)
    
    def __getitem__(self,idx):
        name=self.paths[idx]
        cls_name=name.split("/")[-3]
        cls_idx=self.classes2index[cls_name]
        data=np.load(name)
        data=data[np.newaxis,:]
        sample={"voxel":data,"cls_idx":cls_idx}
        return sample
    
    def __len__(self):
        return len(self.paths)
