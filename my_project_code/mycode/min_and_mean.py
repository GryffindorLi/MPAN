import numpy as np
import os, sys

path1 = '/home/dh/zdd/Lzr/stage3_data_min/train'
path2 = '/home/dh/zdd/Lzr/stage3_data_min/test'

target1 = '/home/dh/zdd/Lzr/stage3_min/train'
target2 = '/home/dh/zdd/Lzr/stage3_min/test'
#os.makedirs(target1)
#os.makedirs(target2)
train = os.listdir(path1)
test = os.listdir(path2)
#train=[204,248,503,1007,1471,2168,2184,2573,2776,2809,2974,3008,3196,3927,4263,4706,6059,6173,6515,7297,7306,7392,7432,7601,8559,8908,9054,9234,9764,10110,10145,10341,11024,11497,11500,12962,13075,13376,13408,13719]
#test=[255,880,990,1286,1553,2262,2813]

tr_f=[str(i)+'.npz' for i in train]
te_f=[str(i)+'.npz' for i in test]
#train_min = [os.path.join(path1, i) for i in train_min]
#test_min = [os.path.join(path2, i) for i in test_min]

for sample in train:
    sam = os.path.join(path1, sample)
    inst = np.load(sam)
    f = inst['feature']
    la = inst['cls']
    sh = f.shape
    if sh[0] == 1:
        f = f.squeeze()
    else:
        f = f.squeeze()
        f = np.min(f, axis=0)
#    f = np.min(f, axis=0)
    np.savez(os.path.join(target1, sample), feature=f, label=la)

for sample in test:
    sam = os.path.join(path2, sample)
    inst = np.load(sam)
    f = inst['feature']
    la = inst['cls']
    sh = f.shape
    if sh[0] == 1:
        f = f.squeeze()
    else:
        f = f.squeeze()
        f = np.min(f, axis=0)
#    f = np.min(f, axis=0)
    np.savez(os.path.join(target2, sample), feature=f, label=la)

target3 = '/home/dh/zdd/Lzr/stage3_mean/train'
target4 = '/home/dh/zdd/Lzr/stage3_mean/test'
#os.makedirs(target3)
#os.makedirs(target4)
'''
for sample in train:
    sam = os.path.join(path1, sample)
    inst = np.load(sam)
    f = inst['feature']
    la = inst['cls']
    sh=f.shape
    if sh[0]==1:
        f=f.squeeze()
    else:
        f = f.squeeze()
        f = np.mean(f, axis=0)
    np.savez(os.path.join(target3, sample), feature=f, label=la)

for sample in test:
    sam = os.path.join(path2, sample)
    inst = np.load(sam)
    f = inst['feature']
    la = inst['cls']
    sh = f.shape
    if sh[0]==1:
        f=f.squeeze()
    else:
        f = f.squeeze()
        f = np.mean(f, axis=0)
    np.savez(os.path.join(target4, sample), feature=f, label=la)
'''