import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from Dataset import ModelNet10VoxelDataset
from voxnet import VoxNet

if __name__=='__main__':
    root="../ModelNet10_voxel/"

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

    def visualize(path):
        fig=plt.figure(figsize=(5,5))
        ax=fig.add_subplot(projection='3d')
        ax.voxels(np.load(path),facecolors='grey', edgecolor='k')
        plt.show()

    # visualize(path="../ModelNet10_voxel/bed/train/bed_000000345.npy")

    # check gpu
    if torch.cuda.is_available():
        if_cuda=True
    else:
        if_cuda=False
    seed_value=0
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED']=str(seed_value)
    if if_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_root = '../ModelNet10_voxel'
    outf = './trained_weights'
    batchSize = 256
    workers =2
    epochs=15

    # training loop
    train_dataset = ModelNet10VoxelDataset(root=data_root, classes=CLASSES, split='train')
    test_dataset = ModelNet10VoxelDataset(root=data_root,classes=CLASSES, split='test')

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=int(workers))
    test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=int(workers))

    voxnet = VoxNet(n_classes=len(CLASSES)) #model instance
    print(voxnet)

    if torch.cuda.is_available():
        voxnet = voxnet.cuda()

    optimizer = optim.Adam(voxnet.parameters(), lr=1e-4) #optimizer

    num_batch=len(train_dataset)
    best_acc=0
    for epoch in range(epochs):
        mean_correct=[]

        for i,sample in enumerate(train_dataloader):
            voxel,cls_idx=sample["voxel"],sample["cls_idx"]
            if torch.cuda.is_available():
                voxel,cls_idx=voxel.cuda(),cls_idx.cude()

            voxel=voxel.float()
            optimizer.zero_grad()
            voxnet=voxnet.train()
            pred=voxnet(voxel)

            loss=F.cross_entropy(pred,cls_idx)
            loss.backward()
            optimizer.step()

            pred_choice=pred.data.max(1)[1]
            correct=pred_choice.eq(cls_idx.data).cpu().sum()
            mean_correct.append(correct.item()/float(batchSize))
        print('\n[%d/%d] train loss: %f mean accuracy: %f' %(epoch+1, epochs,loss.item(), np.mean(mean_correct)))

        # validation
        with torch.no_grad():
            mean_correct=[]
            for j, sample in enumerate(test_dataloader):
                voxel, cls_idx = sample['voxel'], sample['cls_idx']
                if torch.cuda.is_available():
                    voxel, cls_idx = voxel.cuda(), cls_idx.cuda()

                voxel = voxel.float()  
                voxnet = voxnet.eval()
                pred = voxnet(voxel)
                loss = F.nll_loss(pred, cls_idx)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(cls_idx.data).cpu().sum()
                mean_correct.append(correct.item() / float(batchSize))
        acc = np.mean(mean_correct)
        print('test [%d], mean accuracy: %f' % (epoch+1, acc))

        # saving model having best accuracy
        if acc>=best_acc:
            if not os.path.isdir(outf):
                os.mkdir(outf)
            torch.save(voxnet.state_dict(), '%s/weights_%d.pth' % (outf, epoch+1))
            best_acc=acc
