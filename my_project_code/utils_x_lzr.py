# *_*coding:utf-8 *_*
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def show_example(x, y, x_reconstruction, y_pred,save_dir, figname):
    x = x.squeeze().cpu().data.numpy()
    x = x.permute(0,2,1)
    y = y.cpu().data.numpy()
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)
    y_pred = y_pred.cpu().data.numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.savefig(save_dir + figname + '.png')

def save_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, path,modelnet='checkpoint'):
    savepath  = path + '/%s-%f-%04d.pth' % (modelnet,test_accuracy, epoch)
    state = {
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)

def test(model, loader):
    fts = np.zeros([1,256])
    mean_correct = []
    for j, data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _, ft = classifier(points)
        ft = ft.cpu().data.numpy()
        fts = np.concatenate((fts, ft), axis=0)
        pred_choice = pred.data.max(1)[1]     # add here to txt file
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    return np.mean(mean_correct), fts[1:, :]

def test_cls(model, loader):
    mean_correct = []
    for j, data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):
        feature, label = data
        batch_size = len(label)
        feature, label = feature.cuda(), label.cuda()
        classifier = model.eval()
        pred = classifier(feature)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label.long().data).cpu().sum()
        mean_correct.append(100*correct/batch_size)
    return np.mean(mean_correct)

def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_partseg(model, loader, catdict, num_classes = 50,forpointnet2=False):
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    iou_tabel = np.zeros((len(catdict),3))
    iou_list = []
    metrics = defaultdict(lambda:list())
    hist_acc = []
    # mean_correct = []
    temps = np.zeros([1,2048])
    time = str(datetime.datetime.now()).split('.')[0]

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point,_= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points_t = points.transpose(2, 1)
        norm_plt_t = norm_plt.transpose(2, 1)
        points_t, label, target, norm_plt_t = points_t.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt_t.cuda()
        if forpointnet2:
            seg_pred = model(points_t, norm_plt_t, to_categorical(label, 16))
        else:
            labels_pred, seg_pred, _ = model(points,to_categorical(label, 16))
            # labels_pred_choice = labels_pred.data.max(1)[1]
            # labels_correct = labels_pred_choice.eq(label.long().data).cpu().sum()
            # mean_correct.append(labels_correct.item() / float(points.size()[0]))
        # print(pred.size())

        p, l, t, n = points.cpu().data.numpy(), label.cpu().data.numpy(), target.cpu().data.numpy(), norm_plt.cpu().data.numpy()
        s_pred = seg_pred.cpu().detach().numpy()

        os.makedirs('/home/dh/zdd/Lzr/test_data/'+time+'/'+str(batch_id))

        np.save('/home/dh/zdd/Lzr/test_data/'+time+'/'+str(batch_id)+'/s_pred.npy', s_pred)

        np.save('/home/dh/zdd/Lzr/test_data/'+time+'/'+str(batch_id)+'/p.npy', p)

        np.save('/home/dh/zdd/Lzr/test_data/'+time+'/'+str(batch_id)+'/n.npy', n)

        np.save('/home/dh/zdd/Lzr/test_data/'+time+'/'+str(batch_id)+'/t.npy', t)

        np.save('/home/dh/zdd/Lzr/test_data/'+time+'/'+str(batch_id)+'/l.npy', l)

        iou_tabel, iou = compute_cat_iou(seg_pred,target,iou_tabel)
        iou_list+=iou
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        
        temp = pred_choice.view(-1,2048).cpu().data.numpy()
        temps = np.concatenate((temps, temp), axis=0)
        # print('temp:',temps, temps.shape)

        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['inctance_avg_iou'] = np.mean(iou_list)
    # metrics['label_accuracy'] = np.mean(mean_correct)
    iou_tabel = pd.DataFrame(iou_tabel, columns=['iou', 'count', 'mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict))]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    metrics['class_avg_iou'] = np.mean(cat_iou)

    return metrics, hist_acc, cat_iou, temps[1:, :]

def test_semseg(model, loader, catdict, num_classes = 13, pointnet2=False):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        if pointnet2:
            pred = model(points[:, :3, :], points[:, 3:, :])
        else:
            pred, _ = model(points)
        # print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean

def plot_loss_curve(history,n_points_avg,n_points_plot,save_dir):
    curve = np.asarray(history['loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-g')

    curve = np.asarray(history['margin_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-b')

    curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-r')

    plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
    plt.savefig(save_dir + '/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M')) + '_total_result.png')
    plt.close()

def plot_acc_curve(total_train_acc,total_test_acc,save_dir):
    plt.plot(total_train_acc, '-b',label = 'train_acc')
    plt.plot(total_test_acc, '-r',label = 'test_acc')
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Accuracy of training and test')
    plt.savefig(save_dir +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'_total_acc.png')
    plt.close()

def show_point_cloud(tuple,seg_label=[],title=None):
    import matplotlib.pyplot as plt
    if seg_label == []:
        x = [x[0] for x in tuple]
        y = [y[1] for y in tuple]
        z = [z[2] for z in tuple]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        category = list(np.unique(seg_label))
        color = ['b','r','g','y','w','b','p']
        ax = plt.subplot(111, projection='3d')
        for categ_index in range(len(category)):
            tuple_seg = tuple[seg_label == category[categ_index]]
            x = [x[0] for x in tuple_seg]
            y = [y[1] for y in tuple_seg]
            z = [z[2] for z in tuple_seg]
            ax.scatter(x, y, z, c=color[categ_index], cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    plt.show()

def element_wise_max(in_list):   #变长参数函数
    in_feat = np.array(in_list)
    return np.max(in_feat, axis=1)

def element_wise_min(in_list):   #变长参数函数
    in_feat = np.array(in_list)
    in_feat = in_feat.squeeze()
    return np.min(in_feat, axis=0)

def element_wise_mean(in_list):
    in_feat = np.array(in_list)
    in_feat = in_feat.squeeze()
    return np.mean(in_feat, axis=0)

def data_seg(root, phase):   # root must be detailed as /home/dh/zdd/Lzr/experiment_data/2019-11-21 08:41:55/
    # sort the list into proper order
    batches = os.listdir(root)
    batches_int = sorted([int(i) for i in batches])
    batches = [str(j) for j in batches_int]

    # path to different .npy file
    points = [os.path.join(root, i, 'p.npy') for i in batches]
    norms = [os.path.join(root, i, 'n.npy') for i in batches]
    labels = [os.path.join(root, i, 'l.npy') for i in batches]
    parts = [os.path.join(root, i, 't.npy') for i in batches]
    results = [os.path.join(root, i, 's_pred.npy') for i in batches]

    # list to save points and norms from the same part of a instance
#    same_part_points = []
#    same_part_norms = []

    # search for the same parts in one instance
    for i in range(len(batches)):
        points_info = np.load(points[i])
        norms_info = np.load(norms[i])
        labels_info = np.load(labels[i])
        results_info = np.load(results[i]).argmax(axis=2)   # select the final results of segmentation
        for j in range(4):   # batchsize is 4
            point = points_info[j, :, :]  # size batchsize*num_points*3(channel)
            norm = norms_info[j, :, :]   # size batchsize*num_points*3(channel)
            label = labels_info[j]   # size 4
            result = results_info[j, :]   # size batchsize*num_points
            seg_parts = list(set(result))

            instance_no = str(4*i + j)   # instance number of the instance
            os.makedirs(os.path.join('/home/dh/zdd/Lzr/instance_seg', str(phase), instance_no))
            for part in seg_parts:
                # list to save points and norms from the same part of a instance
                same_part_points = []
                same_part_norms = []
                for k in range(len(result)):
                    if result[k] == part:
                        same_part_points.append(point[k, :])
                        same_part_norms.append(norm[k, :])
                same_part_points, same_part_norms = np.array(same_part_points), np.array(same_part_norms)
                #same_part = np.concatenate(same_part_points, same_part_norms, axis=1)
                #os.makedirs(os.path.join('/home/dh/zdd/Lzr/instance_seg', str(phase), instance_no))
                np.savez(os.path.join('/home/dh/zdd/Lzr/instance_seg', str(phase), instance_no, str(part)+'_info.npz'),
                        points=same_part_points, norm_plt=same_part_norms, label=label)
                #np.savetxt(os.path.join('/home/dh/zdd/Lzr/instance_seg_'+str(phase), instance_no, 'class.txt'), label)

def to_2048(part):
    p_out = []
#    n_out = []
    num_p = np.size(part, axis=1)
    choice = np.random.choice(num_p, 2048, True)
    for i in choice:
        p_out.append(part[:, i, :])
#        n_out.append(norm[i, :])
    p_out = np.array(p_out)
#    n_out = np.array(n_out)
    return p_out

def form_feat(root):
    fns = os.listdir(root)
    fns = [os.path.join(root, fn) for fn in fns]
    for fn in fns:
        bag = np.load(fn)
        feats = bag['feature']
        cls = bag['cls']
        feat = np.max(feats, axis=0)
        np.savez(fn, feature=feat, label=cls)


'''
def data_divide(in_root):
    data_fns = os.listdir(in_root)
    classes_dic = {}
    for data_fn in data_fns:
        raw_data = np.loadtxt(os.path.join(in_root, data_fn))
        [row, column] = np.size(raw_data)
        num_ins = row/2048
        ins_0, ins_1, ins_2, ins_3 = np.split(raw_data, num_ins, axis = 1)
        np.save(os.path.join(in_root, data_fn.split('.')[0], 'ins_0.txt'), ins_0, fmt="%d", delimiter=" ")
        np.save(os.path.join(in_root, data_fn.split('.')[0], 'ins_1.txt'), ins_1, fmt="%d", delimiter=" ")
        np.save(os.path.join(in_root, data_fn.split('.')[0], 'ins_2.txt'), ins_2, fmt="%d", delimiter=" ")
        np.save(os.path.join(in_root, data_fn.split('.')[0], 'ins_3.txt'), ins_3, fmt="%d", delimiter=" ")
        classes_dic[os.path.join(in_root, data_fn.split('.')[0], 'ins_0.txt')] = ins_0[0][6]
        classes_dic[os.path.join(in_root, data_fn.split('.')[0], 'ins_1.txt')] = ins_1[0][6]
        classes_dic[os.path.join(in_root, data_fn.split('.')[0], 'ins_2.txt')] = ins_2[0][6]
        classes_dic[os.path.join(in_root, data_fn.split('.')[0], 'ins_3.txt')] = ins_3[0][6]
    return classes_dic

        #seperate instance into different part
        for item in classes_dic.items():
            ins_seg_part(item)
'''


