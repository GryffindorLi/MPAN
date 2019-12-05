import numpy as np
import os, sys
from utils_x_lzr import data_seg

train_root = '/home/dh/zdd/Lzr/experiment_data/2019-11-21 08:41:55'
test_root = '/home/dh/zdd/Lzr/test_data/2019-12-03 14:57:47'

data_seg(train_root, phase = 'train')
data_seg(test_root, phase = 'test')