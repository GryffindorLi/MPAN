import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.toDataset import toDataset
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import ex_partseg 
import ex_clf

# =======================
# toDataset()
# 改进toDataset 函数，toDataset 转换成pytorch 网络能够处理的数据
# toData_n 是将没有标签只有数据的包括在里面
# =======================
def separate(l_point,l_part): 
    part_sta = []    # 统计数组
    part_all = []
    for it in l_part:
        part_set = set([int(i) for i in it])
    sta_part = 0
    sta_mode = 0
    for it_po, it_pa in zip(l_point,l_part):     # 取每个module
        
        part_set = set([int(i) for i in it_pa])        
        # 查找出所有相对应的点，使用字典比较方便,注意每次使用后要进行清空
        part_dict = {}
        for it in part_set:
            part_dict[it] = []
        for po, pa in zip(it_po, it_pa):    # 取每个模型中的每一行
            part_dict[pa].append(po)
        len1 = len(part_dict)
        # 这里可以使用 deep copy 进行操作。
        temp =[i for i in part_dict]
        for k in temp:
            if len(part_dict[k])<=10:
                part_dict.pop(k)
                sta_part += 1
            else:
                part_all.append(np.array(part_dict[k]))  
        if len1 != len(part_dict):
            sta_mode += 1
        part_sta.append(len(part_dict))
    print('有%d个模型，%d part舍去'%(sta_mode, sta_part))
    return np.array(part_all),np.array(part_sta)


# =======================
# toDataset()
# 改进toDataset 函数，toDataset 转换成pytorch 网络能够处理的数据
# toData_n 是将没有标签只有数据的包括在里面,就是往里面加入假的标签
# =======================
def toDataset_n(part_all,need=0):
    pass
   # 这里处理的太复杂了，完全没有必要

# =======================
# point_2048
# 将pointnet 扩展为标准的2048个点
# =======================
def point_2048(part_all):
    temp_all = []
    for it in part_all:
        len1 = len(it)
        choice = np.random.choice(range(len1),2048,True)
        temp = it[choice,:]
        temp_all.append(temp)
    return temp_all 

# =======================
# point_2048
# 将pointnet 扩展为标准的2048个点
# =======================
def main():
    
    # =================
    # 引入shapeNet的数据
    # ==================
    TEST_DATASET = PartNormalDataset(npoints=2048, split='test',normalize=False,jitter=False)

    l_point = []
    l_label = []
    l_part = []
    i = 0
    for point,label,part,_ in TEST_DATASET:
        l_point.append(point)
        l_label.append(label)
        l_part.append(part)
        print(label,end=',')
    l_point = np.array(l_point)
    l_label = np.array(l_label)    # label 在这里基本不涉及什么操作，
    l__part = np.array(l_part)
    
    
    # ====================
    # 引入modelnet 的数据
    # ==========================
#     datapath = './data/ModelNet/'
#     train_data, train_label, test_data, test_label = load_data(datapath, classification=True)     
#     l_point = np.array(test_data)
#     l_label = np.array(test_label)


    
    ch = 1 
    ch_all = 1
# --------------------------
    # 各个数据组合进行测试，
    # 1=org-clf             out:feature+label
    # 2=org-part-clf            feature+label+part_sta 
    # 3=org-partseg-clf         feature+label+part_sta
    if ch == 1 :
        print('Org 处理...')
        testDataset = toDataset(l_point)
        fts = ex_clf.main(testDataset)      # 输入的数据类型point,label
        print('运算完毕')

        temp_dict = {}  # feature, label, part_sta
        temp_dict['feature'] = fts
        temp_dict['label'] = l_label.reshape(1,-1)
        savemat('org_shapeNet_test.mat', temp_dict)
        print('org_clf.mat 文件已经保存！')

#        print('保存源文件...')
#        for i,(j,k) in tqdm(enumerate(zip( l_point, l_label), 0), total=len(l_label)):
#            fp = os.path.join('./result/shapeNet/', '%04.d'%i+'_'+'%02.d'%k+'.txt')
#            fo = open(fp, 'w')
#            np.savetxt(fo, np.array(j).astype(np.float32), fmt='%.6f')
#            fo.close()
#        print('保存源文件完成')
    
    elif ch == 2:
        print('Part 处理...')
        part_all,part_sta = separate(l_point, l_part)
        temp_part = part_all
        part_all = point_2048(part_all)
        part_all = toDataset(part_all)
    
        aa = ex_clf.main(part_all)      # 输入的数据类型point,label
        print('运算完毕！')
        temp_dict = {}          # feature, label, part_sta
        temp_dict['feature'] = aa
        temp_dict['label'] = l_label.reshape(1,-1)
        temp_dict['part_sta'] = part_sta
        # savemat('part_clf_shape.mat',temp_dict)
        savemat('part_m.mat', temp_dict)
        print('part_clf.mat 文件已经保存！')

        print('保存part文件...')
        index = 0
        for it1 in range(len(part_sta)):
            for it2 in range(part_sta[it1]):
                # it1, label[it1], it2== 索引，类标签，part 分开
                
                fp = os.path.join('./result/part','%04.d'%it1+'_'+'%02.d'%l_label[it1,0]+'_'+'%01.d'%it2+'.txt')
                fo = open(fp, 'w')
                np.savetxt(fo, np.array(temp_part[index]).astype(np.float32), fmt='%.6f')
                fo.close()
                index += 1
        print('保存part文件完成')

  
    elif ch == 3 or ch_all == 1:
        print('Partseg 处理...')
        part_predict = ex_partseg.main(TEST_DATASET)
        part_all,part_sta = separate(l_point, part_predict)
    
        part_all = point_2048(part_all)
        part_all = toDataset(part_all)
        aa = ex_clf.main(part_all)      # 输入的数据类型point,label
        print('运算完毕！') 
    
        temp_dict = {}  # feature, label, part_sta
        temp_dict['feature'] = aa
        temp_dict['label'] = l_label.reshape(1,-1)
        temp_dict['part_sta'] = part_sta
    
        # savemat('partseg_clf_shape.mat',temp_dict)
        savemat('partseg_m.mat',temp_dict)
        print('partset_clf.mat 文件已经保存！')
    else: print('没有相关的数据！')

    #--------------------
    #* 对函数进行分割
    #--------------------
    # part_predict = ex_partseg.main(temp)
    # part_all,part_sta = separate(l_point, part_predict)

    # 注： 输出的part 的顺序每次都是不同的，因为进行了点扩充    
    # 将数据进行分割，并对相应的类进行统计（每个模型对应技术） 
    
    # 单独的数据操作
    # part_all,part_sta = separate(l_point, l_part)



    # 这是进行分类操作
    # temp_all = point_2048(part_all)
    # part_all = toDataset_n(l_point, 1)
    # aa = ex_clf.main(part_all)      # 输入的数据类型point,label






    

    



    
    
    
if __name__ == '__main__':
    main()

    
