# 科研问题记录

## 11.19中午：

stage_1的result是一个3D array，不明白意思，问一下
解决方法：np.save()，跑完以后print看一下样子

##11.21下午：
segmentation的结果是一个batch_size * num_points * classes的三维矩阵
改成batch_size * num_points的矩阵

p.npy, n.npy是batch_size * num_points * 3
l.npy只有4个数，表示4个的类别
t.npy是batch_size * num_points * classes

##11.21晚上：
数据接口已经重写，testdataset的接口还没写，但是只要修改储存地址即可

##12.3：
FC_train第95行，数据的size不一致

##12.5：

tip：numpy中的savez函数好用

training data 13998个
testing data 2874个

## 12.6：

特征提取模型换成pointnet，pointnet++有input size的问题。

## 12.12

第二阶段跑通，清理一下储存空间，feature跑完以后删除instance_seg

## 12.13

feature已经全部提取出来，只剩下training process了！

## 1.9

第一次训练完成，best accuracy 71%

调参

考虑：feature fusion的改善思路（加入global feature；pooling方式改进）

## 1.11

finish mean-pooling and min-pooling

服务器和PC上代码不一致