# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:52:23 2020

@author: mimif
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from glob import glob
import os
import sys
from collections import Counter
import operator

# from google.colab import files
from model import PointNetCls

#todo: BN, dropout, transform

batch_size = 32

class PointNet(nn.Module):
    def __init__(self, k, pool_size=10000, dropout_rate = 0):
        super(PointNet, self).__init__()
        self.k_ = k
        self.dropout_rate_ = dropout_rate
        self.linear64_1 = nn.Conv1d(3, 64, 1)#nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear64_2 = nn.Conv1d(64, 64, 1) #nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear64_3 = nn.Conv1d(64, 64, 1) #nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear128 = nn.Conv1d(64, 128, 1) #nn.Linear(64, 128)
        self.linear1024 = nn.Conv1d(128, 1024, 1) #nn.Linear(128, 1024)
        self.pool = nn.MaxPool1d(pool_size) #?
        self.linear512 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(p=self.dropout_rate_)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear256 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.lineark = nn.Linear(256, self.k_)
        
        
    
    def forward(self, x):
        #ignore input transform
        # (bs, 10000, 3)
        x = x.transpose(1, 2)
        # (bs, 3, 10000)
        x = torch.nn.ReLU()(self.bn1(self.linear64_1(x)))
        x = torch.nn.ReLU()(self.bn2(self.linear64_2(x)))
        #ignore feature transform
        x = torch.nn.ReLU()(self.bn3(self.linear64_3(x)))
        x = torch.nn.ReLU()(self.linear128(x))
        x = torch.nn.ReLU()(self.linear1024(x))
        # (bs, 1024, 10000)
        x = x.transpose(1, 2)
        # (bs, 10000, 1024)
        #https://stackoverflow.com/questions/59788096/pytorch-apply-pooling-on-specific-dimension
        #do pooling on second dimension
        x = self.pool(x.permute(0,2,1)).permute(0,2,1)
        # (bs, 1, 1024)
        x = torch.squeeze(x, 1)
        # (bs, 1024)
        x = torch.nn.ReLU()(self.bn4(self.linear512(x)))
        if self.dropout_rate_ > 0:
          x = torch.nn.ReLU()(self.bn5(self.dropout1(self.linear256(x))))
        else:
          x = torch.nn.ReLU()(self.bn5(self.linear256(x)))
        # (bs, 256)
        # need log softmax function, not ReLU!
        x = F.log_softmax(self.lineark(x), dim=1)
        # (bs, 40)
        return x
    
class PointNetSmall(nn.Module):
    def __init__(self, k, pool_size=10000, dropout_rate = 0):
        super(PointNetSmall, self).__init__()
        self.k_ = k
        self.dropout_rate_ = dropout_rate
        self.linear8_1 = nn.Conv1d(3, 8, 1)
        self.bn1 = nn.BatchNorm1d(8)
        self.lineara = nn.Conv1d(8, 8, 1)
        self.bna = nn.BatchNorm1d(8)
        self.linearb = nn.Conv1d(8, 8, 1)
        self.bnb = nn.BatchNorm1d(8)
        self.linear16 = nn.Conv1d(8, 64, 1)
        self.pool = nn.MaxPool1d(pool_size)
        self.linear8_2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=self.dropout_rate_)
        self.lineark = nn.Linear(32, self.k_)
    
    def forward(self, x):
        #ignore input transform
        x = x.transpose(1, 2)
        x = torch.nn.ReLU()(self.bn1(self.linear8_1(x)))
        x = torch.nn.ReLU()(self.bna(self.lineara(x)))
        x = torch.nn.ReLU()(self.bnb(self.linearb(x)))
        x = torch.nn.ReLU()(self.linear16(x))
        x = x.transpose(1, 2)
        x = self.pool(x.permute(0,2,1)).permute(0,2,1)
        x = torch.squeeze(x, 1)
        if self.dropout_rate_ > 0:
          x = torch.nn.ReLU()(self.bn2(self.dropout1(self.linear8_2(x))))
        else:
          x = torch.nn.ReLU()(self.bn2(self.linear8_2(x)))
        x = F.log_softmax(self.lineark(x), dim=1)
        return x

class PointNetDropout(nn.Module):
    def __init__(self, k, pool_size=10000):
        super(PointNetDropout, self).__init__()
        self.k_ = k
        self.linear64_1 = nn.Conv1d(3, 64, 1)#nn.Linear(3, 64)
        self.linear64_2 = nn.Conv1d(64, 64, 1) #nn.Linear(64, 64)
        self.linear64_3 = nn.Conv1d(64, 64, 1) #nn.Linear(64, 64)
        self.linear128 = nn.Conv1d(64, 128, 1) #nn.Linear(64, 128)
        self.linear1024 = nn.Conv1d(128, 1024, 1) #nn.Linear(128, 1024)
        self.pool = nn.MaxPool1d(pool_size) #?
        self.linear512 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear256 = nn.Linear(512, 256)
        self.lineark = nn.Linear(256, self.k_)
    
    def forward(self, x):
        #ignore input transform
        # (bs, 10000, 3)
        x = x.transpose(1, 2)
        # (bs, 3, 10000)
        x = torch.nn.ReLU()(self.linear64_1(x))
        x = torch.nn.ReLU()(self.linear64_2(x))
        #ignore feature transform
        x = torch.nn.ReLU()(self.linear64_3(x))
        x = torch.nn.ReLU()(self.linear128(x))
        x = torch.nn.ReLU()(self.linear1024(x))
        # (bs, 1024, 10000)
        x = x.transpose(1, 2)
        # (bs, 10000, 1024)
        #https://stackoverflow.com/questions/59788096/pytorch-apply-pooling-on-specific-dimension
        #do pooling on second dimension
        x = self.pool(x.permute(0,2,1)).permute(0,2,1)
        # (bs, 1, 1024)
        x = torch.squeeze(x, 1)
        # (bs, 1024)
        x = torch.nn.ReLU()(self.dropout1(self.linear512(x)))
        x = torch.nn.ReLU()(self.linear256(x))
        # (bs, 256)
        # need log softmax function, not ReLU!
        x = F.log_softmax(self.lineark(x), dim=1)
        # (bs, 40)
        return x

class ModelNetDataset(Dataset):
    def __init__(self, root_dir, transform = None, split='train'):
        self.root_dir = root_dir
        #vehicle / pedestrian / cyclist / other
        self.split = split
        self.fnames = []
        with open(os.path.join(self.root_dir, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fnames.append(
                    os.path.join(self.root_dir, line.strip().rsplit('_', 1)[0] + '/' + line.strip() + '.txt'))
        
        classnames = []
        for fname in self.fnames:
            classnames.append(
                fname[fname.rfind('/',0,fname.rfind('/'))+1:fname.rfind('/')])
        classnames_uniq = sorted(list(set(classnames)))
        print("unique class names", len(classnames_uniq), classnames_uniq)
        classname2classid = dict(zip(classnames_uniq, 
                                     list(range(len(classnames_uniq)))))
        self.classes = []
        for classname in classnames:
            self.classes.append(classname2classid[classname])
        # print("classes", self.classes[:10])
            
        self.transform = transform
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        cloud = np.loadtxt(self.fnames[idx], delimiter=',')
        #discard normals
        cloud = cloud[:, :3]
        # cloud.astype(np.double)
        _class = self.classes[idx]
        
        sample = {'cloud': cloud, 'class': _class}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform = None, split='train'):
        self.root_dir = root_dir
        self.split = split
        """
        according to object3d.py's cls_type_to_id:
        it will be {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4},
        other classes will be -1
        """
        self.classnames = ['Other', 'Vehicle', 'Pedestrian', 'Cyclist']
        self.classmap = {-1:0,1:1,2:2,3:3,4:1}
        
        self.ids = []
        self.classes = []
        with open(os.path.join(self.root_dir, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                # print(line)
                # print(int(line.strip().rsplit('_')[-1]))
                self.ids.append(line.strip())
                self.classes.append(
                    self.classmap[int(line.strip().rsplit('_')[-1])])
        self.transform = transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        cloud = np.loadtxt(
            os.path.join(self.root_dir, self.ids[idx] + '.txt'), delimiter=' ')
        #discard normals
        cloud = cloud[:, :3]
        # cloud.astype(np.double)
        _class = self.classes[idx]
        
        sample = {'cloud': cloud, 'class': _class, 'id': self.ids[idx]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class RotateFromY(object):
    """
    RotateFromY
    
    the height direction of KITTI dataset is y,
    need to change it to z so we can use RandomRotateOverZ
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        cloud[:,1], cloud[:,2] = cloud[:,2], -cloud[:,1]
        # np.savetxt("D:/preprocessed/"+_id.rsplit('/')[-1]+'_rfy.txt', cloud)
        return {'cloud': cloud, 'class': _class, 'id': _id}

class RandomRotateOverZ(object):
    """
    RandomRotateOverZ
    """

    def __init__(self):
        theta = np.random.uniform(-np.pi,np.pi)
        ux, uy, uz = 0, 0, 1
        cost = np.cos(theta)
        sint = np.sin(theta)
        #https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        self.rot_mat = np.matrix([
          [cost+ux*ux*(1-cost),  ux*uy*(1-cost)-uz*sint, ux*uz*(1-cost)+uy*sint],
          [uy*ux*(1-cost)+uz*sint, cost+uy*uy*(1-cost),  uy*uz*(1-cost)-ux*sint],
          [uz*ux*(1-cost)-uy*sint, uz*uy*(1-cost)+ux*sint, cost+uz*uz*(1-cost)]])
        pass

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        #print("cloud", cloud.shape)
        cloud = np.matmul(self.rot_mat, cloud.T)
        cloud = cloud.T
        #print("cloud", cloud.shape)
        # np.savetxt("D:/preprocessed/"+_id.rsplit('/')[-1]+'_rroz.txt', cloud)
        return {'cloud': cloud, 'class': _class, 'id': _id}

class AddGaussianNoise(object):
    """
    AddGaussianNoise
    """

    def __init__(self, amp = 0.2):
        self.amp_ = amp

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        cloud = cloud + np.random.normal(loc=0,scale=self.amp_,size=cloud.shape)
        # np.savetxt("D:/preprocessed/"+_id.rsplit('/')[-1]+'_gauss.txt', cloud)
        return {'cloud': cloud, 'class': _class, 'id': _id}

class RandomCrop(object):
    """
    RandomCrop
    """

    def __init__(self, maxratio = 0.5, ts = 1000):
        self.maxratio_ = maxratio
        self.ts_ = ts

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        zmax = np.max(cloud,axis=0)[2]
        zmin = np.min(cloud,axis=0)[2]
        zrange = zmax-zmin
        ratio = np.random.uniform(0, self.maxratio_)
        newzmin = zmin + zrange*ratio
        cloud_crop, cloud_remain = cloud[cloud[:,2]>=newzmin,:], cloud[cloud[:,2]<newzmin,:]
        if cloud_crop.shape[0] < self.ts_:
          cloud_remain = cloud_remain[np.random.choice(cloud_remain.shape[0], self.ts_-cloud_crop.shape[0], replace=False), :]
          cloud_crop = np.vstack([cloud_crop, cloud_remain])
        #np.savetxt("D:/preprocessed/"+_id.rsplit('/')[-1]+'_crop.txt', cloud)
        return {'cloud': cloud_crop, 'class': _class, 'id': _id}

class InputDropout(object):
    """
    InputDropout
    """

    def __init__(self, ts = 1000):
        self.ts_ = ts# target size
        # print("target size", self.ts_s)

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        cloud = cloud[np.random.choice(cloud.shape[0], self.ts_, replace=False), :]
        #np.savetxt("D:/preprocessed/"+_id.rsplit('/')[-1]+'_resample.txt', cloud)
        return {'cloud': cloud, 'class': _class, 'id': _id}

class Normalize(object):
    """
    normalize to [-0.5, 0.5]
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        cloud = np.reshape(cloud, (-1,3))
        lower = np.min(cloud, axis=0)
        upper = np.max(cloud, axis=0)
        center = (lower+upper)/2.0
        # move to (0,0,0)
        cloud = cloud - center
        # resize to (-0.5, 0.5)
        ratio = 1.0/(upper - lower).max()
        cloud = cloud * ratio
        #np.savetxt("D:/preprocessed/"+_id.rsplit('/')[-1]+'_normalize.txt', cloud)
        return {'cloud': cloud, 'class': _class, 'id': _id}

class ToTensor(object):
    """
    Convert numpy array to torch tensor
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        cloud, _class, _id = sample['cloud'], sample['class'], sample['id']
        cloud = torch.from_numpy(cloud)
        return {'cloud': cloud, 'class': _class, 'id': _id}

def make_weights_for_balanced_classes(classes, nclasses):                        
    count = [0] * nclasses
    for _class in classes:
        # print(fname)
        # print(fname[:-4].rsplit('_'))
        # _class = int(fname[:-4].rsplit('_')[-1])-1
        # count[fname[1]] += 1
        count[_class] += 1
    print("count of different classes", count)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(classes)
    for idx, _class in enumerate(classes):
        # _class = int(fname[:-4].rsplit('_')[-1])-1
        weight[idx] = weight_per_class[_class]
    return weight

def evaluate(root_dir="D:/", model_epoch=0):
    test_dataset = KITTIDataset(
            root_dir=root_dir,
            split='kitti_val',
            transform = torchvision.transforms.Compose([
                InputDropout(),
                Normalize(),
                RotateFromY(),
                ToTensor()
                ]))
        
    nworkers = 1

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    
    pn = PointNetSmall(len(test_dataset.classnames), pool_size=1000, dropout_rate=0.5)
    pn.load_state_dict(torch.load('mypn_%d.pth' % (model_epoch)))
    #.float(): RuntimeError: expected scalar type Double but found Float
    pn = pn.float()
    pn = pn.to(device)
    pn.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    #https://stackoverflow.com/questions/50052295/how-do-you-load-mnist-images-into-pytorch-dataloader
    num_batch = int(len(test_dataset) / batch_size)
    total_correct = 0
    total_loss = 0
    
    for batch in range(num_batch):
        j, data = next(enumerate(test_loader, 0))
        points = data['cloud'].float().to(device)
        target = data['class'].to(device)
        with torch.no_grad():
            pred = pn(points)
        val_loss = criterion(pred, target)
        pred_choice = pred.data.max(1)[1]
        # print(target.data,"->",pred_choice)
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct
        total_loss += val_loss
    print("acc", total_correct/len(test_dataset))
    print("loss", total_loss/len(test_dataset))

def evaluate_one(pn, cloud, device=torch.device('cpu')):
    sample = {'cloud': cloud, 'class': -1, 'id': -1}
    
    transform = torchvision.transforms.Compose([
        # InputDropout(),
        Normalize(),
        RotateFromY(),
        # ToTensor()
        ])
    
    sample = transform(sample)
    points = sample["cloud"][np.newaxis,...]
    # print("points", points.shape)
    points = torch.from_numpy(points)
    points = points.float().to(device)
    with torch.no_grad():
        pred = pn(points)
    pred = pred.cpu().detach().numpy()[0] #batch size is 1
    # print("pred", pred)
    pred_choice = np.argmax(pred)
    # because we are using np.log_softmax in PointNet
    # so here we need to use np.exp to convert it back to score
    score = np.exp(pred[pred_choice])
    # print("score", score)
    
    # class, score
    return pred_choice, score
    

if __name__ == "__main__":
    # Set seed
    torch.manual_seed(0)
    
    if torch.cuda.is_available():  
        dev = "cuda:0"
    else:  
        dev = "cpu"  

    device = torch.device(dev)
    print("device", device)

    transform = None
    # random rotation over z-axis
    
    dataset = "kitti" # or "modelnet40"
    
    if dataset == "modelnet40":
        root_dir = "modelnet40_normal_resampled/"
        """
        full_dataset = ModelNetDataset(
            root_dir,
            transform = torchvision.transforms.Compose([
                RandomRotateOverZ(),
                ToTensor()
                ]))
        
        #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size])
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            #num_workers=2,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            #num_workers=2,
            shuffle=False
        )
        """
        
        train_dataset = ModelNetDataset(
            root_dir=root_dir,
            split='modelnet40_train',
            transform = torchvision.transforms.Compose([
                RandomRotateOverZ(),
                ToTensor()
                ]))
        
        test_dataset = ModelNetDataset(
            root_dir=root_dir,
            split='modelnet40_test',
            transform = torchvision.transforms.Compose([
                ToTensor()
                ]))
    elif dataset == "kitti":
        root_dir = "D:/"
        
        train_dataset = KITTIDataset(
            root_dir=root_dir,
            split='kitti_train',
            transform = torchvision.transforms.Compose([
                RotateFromY(),
                RandomCrop(),
                InputDropout(),
                Normalize(),
                RandomRotateOverZ(),
                AddGaussianNoise(),
                ToTensor()
                ]))
        
        test_dataset = KITTIDataset(
            root_dir=root_dir,
            split='kitti_val',
            transform = torchvision.transforms.Compose([
                RotateFromY(),
                InputDropout(),
                Normalize(),
                ToTensor()
                ]))
        
    nworkers = 1
                                                                              
    # For unbalanced dataset we create a weighted sampler
    print("train_dataset.classes", len(train_dataset.classes), train_dataset.classes[:5])
    print("classnames", train_dataset.classnames)
    weights = make_weights_for_balanced_classes(train_dataset.classes, 
                                                len(train_dataset.classnames))
    weights = torch.FloatTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    #https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    #ValueError: sampler option is mutually exclusive with shuffle
    #https://www.cnblogs.com/zmbreathing/p/pyTorch_BN_error.html
    #ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=nworkers,                              
        sampler = sampler, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    
    #pn = PointNet(len(train_dataset.classnames), pool_size=1000, dropout_rate=0.5)
    pn = PointNetSmall(len(train_dataset.classnames), pool_size=1000, dropout_rate=0.1)
    #pn = PointNetDropout(len(train_dataset.classnames), pool_size=1000)
    
    transfer = False
    if transfer:
        pn = PointNetCls(k=len(train_dataset.classnames), feature_transform=False)
        pretrained_dict = torch.load("cls_model_9.pth", map_location=device)
        model_dict = pn.state_dict()
    
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() \
                           if k in model_dict and \
                               model_dict[k].shape == pretrained_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        pn.load_state_dict(model_dict)
        
        for name, param in pn.named_parameters():
            if "feat" in name:
                param.requires_grad = False
    
    #.float(): RuntimeError: expected scalar type Double but found Float
    pn = pn.float()
    pn = pn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pn.parameters(), lr = 1e-2, momentum = 0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=10, min_lr=1e-4, verbose=True)
    
    # optimizer = optim.Adam(pn.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    #https://stackoverflow.com/questions/50052295/how-do-you-load-mnist-images-into-pytorch-dataloader
    iter_idx = 0
    report_iter = 10 #2000
    num_batch = len(train_dataset) / batch_size
    blue = lambda x: '\033[94m' + x + '\033[0m'
    num_epoch = 100
    save_dir = "small/" #"drive/MyDrive/models/"
    fname = save_dir + "train_log.txt"
    for epoch in range(num_epoch):
        running_loss = 0.0
        
        f = open(fname, "a")

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            #RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type 
            #    (torch.cuda.FloatTensor) should be the same
            #.float(): RuntimeError: expected scalar type Double but found Float
            points = data['cloud'].float().to(device)
            target = data['class'].to(device)
            
            # print("cloud", points.shape)
            # print("class", target)
            
            if transfer:
                points = points.permute(0,2,1)
                
            pred = pn(points)
            
            if transfer:
                pred = pred[0]
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            #print("pred", pred.shape)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            line = '[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(batch_size))
            cnter = Counter(data['class'].data.cpu().detach().numpy())
            line += ' ' + str([cnter[i] for i in range(len(train_dataset.classnames))])
            cnter = Counter(pred_choice.cpu().detach().numpy().flatten())
            line += ' -> ' + str([cnter[i] for i in range(len(train_dataset.classnames))])
            print(line)
            f.write(line+"\n")
            
            running_loss += loss.item()
            #if i == 4: break
            if iter_idx % 10 == 0:
                pn.eval()
                j, data = next(enumerate(test_loader, 0))
                points = data['cloud'].float().to(device)
                target = data['class'].to(device)
                if transfer:
                    points = points.permute(0,2,1)
                pred = pn(points)
                if transfer:
                    pred = pred[0]
                val_loss = criterion(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                line = '[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'test', val_loss.item(), correct.item()/float(batch_size))
                cnter = Counter(data['class'].data.cpu().detach().numpy())
                line += ' ' + str([cnter[i] for i in range(len(train_dataset.classnames))])
                cnter = Counter(pred_choice.cpu().detach().numpy().flatten())
                line += ' -> ' + str([cnter[i] for i in range(len(train_dataset.classnames))])
                print(blue(line))
                f.write(line+"\n")
                f.flush()
                f.close()
                f = open(fname, "a")
                torch.save(pn.state_dict(), save_dir + 'mypn_%d.pth' % (iter_idx))
                pn.train()
                # files.download('mypn_%d.pth' % (iter_idx))
                # files.download("train_log.txt")
            """
            if iter_idx % report_iter == report_iter-1:
                #calculate validation loss
                #https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
                val_loss = 0.0
                val_correct = 0
                for sample_batched in val_loader:
                    outputs = pn(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_dataset)
                scheduler.step(val_loss)
                val_acc = 100 * val_correct / len(val_dataset)
                #calculate validation loss end
                
                print('[%d, %5d] loss: %.3f val loss: %.3f val acc: %.3f' %
                      (epoch + 1, iter_idx + 1, running_loss / report_iter,
                       val_loss, val_acc))
                
                running_loss = 0.0
            iter_idx += 1
            """
            iter_idx += 1
        f.flush()
        f.close()
        scheduler.step(running_loss)

    print('Finished Training')