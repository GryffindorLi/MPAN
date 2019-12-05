import os
import sys
import logging
from torch.utils.data import Dataset
import json
import warnings
from utils_x_lzr import to_categorical
from collections import defaultdict
from data_utils.ShapeNetDataLoader import PartNormalDataset
import datetime
from utils_x_lzr import test_partseg
from tqdm import tqdm
import numpy as np

class seg_loader(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=True, jitter=False):
        self.npoints = npoints
        self.root = '../Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize
        self.jitter = jitter

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        # print(self.cat)

        self.meta = {}  # 获取训练集，验证集和测试集的索引
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:   # item 为数据集中的物品。
            # print('category', item)
            self.meta[item] = []    #将对应的txt存入一个列表中
            dir_point = os.path.join(self.root, self.cat[item])  #获取所有TXT文件
            fns = sorted(os.listdir(dir_point))  #排序，fns is the abbreviation of 'filenames'
            # print(fns[0][0:-4])
            if split == 'trainval':   #选取读取的数据集
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])  # os.path.basename返回路径中的最终文件名，os.path.splitext分离文件名和拓展名
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))  #元组保存文件名以及对应标签

        self.classes = dict(zip(self.cat, range(len(self.cat))))  # label for classification task.
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}  #label for part segmentation task.

        for cat in sorted(self.seg_classes.keys()):
            print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)  # astype covert data into another data form.
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:  #send data into cache in order to speed up I/O.
                self.cache[index] = (point_set, normal, seg, cls)
        if self.normalize:
            point_set = pc_normalize(point_set)
        if self.jitter:
            jitter_point_cloud(point_set)
        choice = np.random.choice(len(seg), self.npoints, replace=True)  #randomly choose data for training or testing.
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]
#        with open('../stage1_data_'+str(datetime.datetime.now()).split('.')[0]+'/input_point_'+str(cls)+str(fn)+'.txt', 'w') as f:
#            for i in range(len(point_set)):
#                f.write(str(point_set[i, :])+' '+str(normal[i,:])+' '+seg[i]+'/n')
         #   f.write(str(point_set) + ' ' + str(normal) + ' ' +str(seg))
        return point_set, cls, seg, normal

class parts_loader(Dataset):
    def __init__(self, root):
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.root = root
        items = os.listdir(self.root)
        items = [os.path.join(self.root, item) for item in items]
        part_dict = {}
        instances = []
        for item in items:
            parts = os.listdir(item)
            part_dict[item] = parts
            instances.append(parts)

        self.part_dict = part_dict
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]

        # define list for traverse the list and send different parts into different net
        points_sets = []
        norm_sets = []
        labels = []

        for part_fn in instance:
            part = np.load(part_fn)
            points = part['points']
            norms = part['norm_plt']
            label = part['label']
            labels.append(label)
            points_sets.append(points)
            norm_sets.append(norms)

        return points_sets, norm_sets, labels[0]

'''
class parts_loader(Dataset):
    def __init__(self, root):   # root must be detailed as /home/dh/zdd/Lzr/experiment_data/2019-11-21 08:41:55/
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.root = root
        batches = os.listdir(root)
        batches_int = sorted([int(i) for i in batches])
        batches_sort = [str(j) for j in batches_int]
        self.batches = batches_sort

        points = [os.path.join(root, i, 'p.npy') for i in self.batches]
        norms = [os.path.join(root, i, 'n.npy') for i in self.batches]
        labels = [os.path.join(root, i, 'l.npy') for i in self.batches]
        parts = [os.path.join(root, i, 't.npy') for i in self.batches]
        results = [os.path.join(root, i, 's_pred.npy') for i in self.batches]



    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        batch = self.batches[item]
        points = np.load(self.points[item])
        norms = np.load(self.norms[item])
        pred_parts = np.load(self.results[item]).argmax(2)
        labels = np.load(self.labels[item])

        return points, norms, pred_parts, labels
'''

class FC_input_loader(Dataset):
    def __init__(self, root):   # root is the place that saves the feature vector
        self.root = root

        batches = os.listdir(self.root)
        self.batches = [os.path.join(self.root, i) for i in batches]
        features = []
        for batch in self.batches:
            fns = os.listdir(batch)
            for fn in fns:
                features.append(os.path.join(batch, fn))
        self.features = features

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        feat = self.features[idx]
        cls = feat.split('/')[-1]
        cls = cls.split('.')[0]
        feature = np.load(feat)
        return feature, cls

'''
class feature_extract_loader(Dataset):
    def __init__(self, root, split='train', normalize=True, jitter = False):
        self.root = root
        self.split = split
        self.normalize = normalize
        self.jitter = jitter
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.ins2part = {}
        self.classes = {}
        self.data = []
        batches = os.listdir(self.root)
        for batch in batches:
            instances = os.listdir(batch)
            for instance in instances:
                ins_data = np.loadtxt(instance)
                self.classes[instance] = ins_data[0, 6]

                num_part = len(set(ins_data[:][-1]))

                self.data.append((instance, ins_data[0, 6], num_part))

    def __getitem__(self, idx):
        fn = self.data[idx]
#        cls = self.data[idx][1]
#        cls = np.array(cls).astype(np.int32)
        n_parts = self.data[idx][1]
        data = np.loadtxt(fn[0]).astype(np.float32)
        pointset = data[:, 0:3]
        norm = data[:, 3:6]
        cls = data[:, 6].astype(np.int32)
        if self.normalize:
            pointset = pc_normalize(pointset)
        if self.jitter:
            jitter_point_cloud(pointset)
        return pointset, norm, cls, n_parts

#        for i in range(n_parts):
#            name = 'part' + str(i)
#            locals()[name] = np.split(np.loadtxt(fn[0]).astype(np.float32), n_parts, dim = 1)[i]


    def __len__(self):
        return len(self.ins2part)
'''
