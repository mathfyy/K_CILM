# -*- coding: utf-8 -*-
"""
"""
# License: BSD
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
import random

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import evaluation_metrics
import scipy
import scipy.io

from resnet3D import resnet101, cscBlock

import pdb

from torch.nn import Parameter
import pickle
from skimage import io, transform
import SimpleITK as sitk

img_size = 224
trial = 2
randseed = trial
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.cuda.manual_seed_all(randseed)

class BinaryCrossEntropyLoss_aw(nn.Module):
    def __init__(self, ):
        super(BinaryCrossEntropyLoss_aw, self).__init__()

    def forward(self, input, target, alpha=1.0):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        epsilon = 1.e-9
        multi_hot_key = target
        logits = input
        zero_hot_key = 1 - multi_hot_key
        loss = -alpha * multi_hot_key * (1 - logits) * (logits + epsilon).log()
        loss += -(1 - alpha) * zero_hot_key * logits * (1 - logits + epsilon).log()
        return loss.nanmean()


class CrossEntropyLoss_aw(nn.Module):
    def __init__(self, ):
        super(CrossEntropyLoss_aw, self).__init__()

    def forward(self, input, target, alpha=1.0):
        epsilon = 1.e-9
        # loss = -alpha * (target * input.log() + (1 - target) * (-input).log())
        loss = -alpha * (target * (input + epsilon).log())
        return loss.nanmean()


def default_loader(dir):
    itkimage = sitk.ReadImage(dir)  # W,H,D  图像
    origin = itkimage.GetOrigin()  # 原点坐标 x, y, z
    spacing = itkimage.GetSpacing()  # 像素间隔 x, y, z
    direction = itkimage.GetDirection()  # 图像方向
    if np.any(direction != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):  # 判断是否相等
        isflip = True
    else:
        isflip = False
    img_array = sitk.GetArrayFromImage(itkimage)  # D,H,W 数组
    img_array = img_array.transpose(2, 1, 0)  # W,H,D  数组
    if (isflip == True):
        img_array = img_array[:, ::-1, ::-1]  #::-1 倒序
    #    print(img_array,origin,spacing,direction)
    return img_array, origin, spacing, direction


def writeArrayToNii(img, savePath, name):
    img = img.to("cpu")
    img = img.detach()
    result = sitk.GetImageFromArray(img)
    sitk.WriteImage(result, os.path.join(savePath, name))
    return result


class load3DDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file, header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx, 0])
        img0 = default_loader(img_name + '_T1.nii')
        imgT1 = img0[0].copy().astype(np.float32)
        img1 = default_loader(img_name + '_T2.nii')
        imgT2 = img1[0].copy().astype(np.float32)
        img2 = default_loader(img_name + '_T1CE.nii')
        imgT1ce = img2[0].copy().astype(np.float32)
        img3 = default_loader(img_name + '_FL.nii')
        imgFL = img3[0].copy().astype(np.float32)

        # img4 = default_loader(img_name + '_brainMask.nii')
        # brainMask = img4[0].copy().astype(np.float32)
        # imgT1 = np.multiply(imgT1, brainMask > 0)
        # imgT2 = np.multiply(imgT2, brainMask > 0)
        # imgT1ce = np.multiply(imgT1ce, brainMask > 0)
        # imgFL = np.multiply(imgFL, brainMask > 0)

        image = np.concatenate((imgT1, imgT2, imgT1ce, imgFL), axis=2).transpose((2, 0, 1))

        # writeArrayToNii(torch.tensor(image),
        #                 '/data1/DataSet/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorGene/CL_gene_data/',
        #                 'image.nii')

        label = self.labels.iloc[idx, 1:].values
        image_id = self.labels.iloc[idx, :].values
        label = label.astype('double')

        # 标签反转
        # label[label == 1] = 10
        # label[label == 0] = 1
        # label[label == 10] = 0

        if self.transform:
            image = self.transform(image)
        return image_id[0], image, label


# trainval需要水平翻转，val与test不需要水平翻转
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ])
}


def train_model(CNN, csc,Ag_param, class_token, criterion, distill_criterion,
                maxEntropy_criterion, optimizer_CNN,
                optimizer_csc,
                task_id, label_num, pre_Ag_param, pre_CNN=None, pre_classifier=None, pre_csc=None,
                pre_class_token=None,
                num_epochs=1, init_lr=0.001):
    sizeD = 28
    factor = 0.1
    lr_scheduler_eps = 1e-8
    lr_scheduler_patience = 3
    scheduler_CNN = lr_scheduler.ReduceLROnPlateau(optimizer_CNN, mode='min', factor=factor,
                                                   patience=lr_scheduler_patience,
                                                   verbose=True, threshold=lr_scheduler_eps,
                                                   threshold_mode="abs")
    scheduler_csc = lr_scheduler.ReduceLROnPlateau(optimizer_csc, mode='min', factor=factor,
                                                    patience=lr_scheduler_patience,
                                                    verbose=True, threshold=lr_scheduler_eps,
                                                    threshold_mode="abs")
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('Current learning rate: ' + '%.8f' % init_lr)
        CNN.train()
        csc.train()
        running_loss = 0.0
        for index, (image_ids, inputs, labels) in enumerate(train_loader):
            # if index > 0:
            #     break

            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.float()
            labels = labels.float()
            labels = labels[:, 0:label_num]
            sizeD = 28
            inputs = inputs.permute(0, 2, 1, 3)
            input = torch.unsqueeze(inputs, dim=1)
            inputs = torch.cat((input[:, :, 0:sizeD, :, :], input[:, :, sizeD:2 * sizeD, :, :],
                                input[:, :, 2 * sizeD:3 * sizeD, :, :],
                                input[:, :, 3 * sizeD:4 * sizeD, :, :]), dim=1)

            #
            # writeArrayToNii(inputs[0, 0, :, :, :], data_path, 'inputs.nii')

            # zero the parameter gradients
            optimizer_CNN.zero_grad()
            optimizer_csc.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                class_token = nn.Parameter(class_token)
                Ag_param = nn.Parameter(Ag_param)
                feature = CNN(inputs)
                F = feature["attentions"][3]
                output = csc(class_token, F, Ag_param)

                # 计算当前batchsize中，不同标签的正样本占比,作为损失函数的权重
                loss = torch.tensor(0.0)
                multi_w = torch.zeros(size=(1, labels.shape[1]))
                for ic in range(labels.shape[1]):
                    pos_num = len(torch.masked_select(labels[:, ic], labels[:, ic] == 1))
                    neg_num = len(torch.masked_select(labels[:, ic], labels[:, ic] == 0))

                    if pos_num + neg_num == 0:
                        multi_w[0, ic] = -1
                    else:
                        multi_w[0, ic] = neg_num / (pos_num + neg_num)

                    tOut = torch.masked_select(output[:, ic], labels[:, ic] != -1)
                    tlab = torch.masked_select(labels[:, ic], labels[:, ic] != -1)
                    if tlab.shape[0] > 0:
                        loss = loss + criterion(tOut.unsqueeze(1), tlab.unsqueeze(1), multi_w[0, ic])

                if loss <= 0:
                    continue
                ########################################################################################
                # 计算蒸馏损失
                ########################################################################################
                if (task_id > 0):
                    pre_feature = pre_CNN(inputs)
                    pre_F = pre_feature["attentions"][3]
                    pre_output = pre_csc(pre_class_token, pre_F, pre_Ag_param)
                    # distill_loss = distill_criterion(output[:, 0:(task_id) * num_classes], pre_output)
                    maxEntropy_loss = maxEntropy_criterion(output[:, 0:(task_id) * num_classes],
                                                           output[:, 0:(task_id) * num_classes])

                    # 优化蒸馏损失
                    thr = 0.5
                    E = np.array([1.0/3.0, 2.0/3.0, 1])
                    data = np.array([[E[1], E[2], E[0], E[1]], [E[1], E[0], E[2], E[1]],
                                     [E[1], E[0], E[0], E[1]]])
                    G = torch.from_numpy(data)
                    multi_w_kl = torch.zeros(size=(pre_output.shape[0], pre_output.shape[1]))
                    lambdas = 0.5
                    for ic in range(pre_output.shape[1]):
                        lack_w = len(torch.masked_select(labels[:, ic], labels[:, ic] == -1))
                        pos_w = len(torch.masked_select(labels[:, ic], labels[:, ic] == 1))
                        neg_w = len(torch.masked_select(labels[:, ic], labels[:, ic] == 0))
                        R = torch.from_numpy(np.array([1.0, 1.0, 1.0])).unsqueeze(0)
                        if neg_w + pos_w > 0:
                            R = torch.from_numpy(
                                np.array([(neg_w / (neg_w + pos_w)),(pos_w / (neg_w + pos_w)),
                                          1.0])).unsqueeze(0)

                        S = torch.zeros(size=(3, 4))
                        for ib in range(pre_output.shape[0]):
                            if labels[ib, ic] == 0 or labels[ib, ic] == 1:
                                r_loc = labels[ib, ic].to(torch.long)
                            else:
                                r_loc = 2
                            if pre_output[ib, ic] > thr:
                                if output[ib, ic] > thr:
                                    S[r_loc, 0] = S[r_loc, 0] + 1.0
                                    c_loc = 0
                                else:
                                    S[r_loc, 1] = S[r_loc, 1] + 1.0
                                    c_loc = 1
                            else:
                                if output[ib, ic] > thr:
                                    S[r_loc, 2] = S[r_loc, 2] + 1.0
                                    c_loc = 2
                                else:
                                    S[r_loc, 3] = S[r_loc, 3] + 1.0
                                    c_loc = 3
                            multi_w_kl[ib, ic] = R[0, r_loc] * G[r_loc, c_loc] * lambdas

                    pre_output_ = pre_output + (output[:, 0:(task_id) * num_classes] - pre_output) * multi_w_kl.to(
                        device)
                    distill_loss = distill_criterion(output[:, 0:(task_id) * num_classes], pre_output_)

                ########################################################################################
                if task_id > 0:
                    a = 0.15
                    b = 4e-3
                    loss = a * loss + (1 - a) * distill_loss - maxEntropy_loss * b

                loss.backward(retain_graph=True)
                print('{} CurLoss: {:.4f}'.format(
                    'CurTrain', loss))
                # 获取更新后的Ag
                cur_lr=init_lr
                for param_group in optimizer_csc.param_groups:
                    cur_lr = param_group['lr']
                class_token = class_token - cur_lr * class_token.grad
                pickle.dump({'class_token': class_token},
                            open(data_path + '/result/K_CILM/class_token' + str(task_id) + '.pkl', 'wb'),
                            pickle.HIGHEST_PROTOCOL)

                Ag_param = Ag_param - cur_lr * Ag_param.grad
                Ag_param[0, 0] = min(1.0, max(0.0001, Ag_param[0, 0]))
                Ag_param[0, 1] = min(1.0, max(0.0001, Ag_param[0, 1]))
                Ag_param[0, 2] = min(1.0, max(0.0001, Ag_param[0, 2]))
                pickle.dump({'Ag_param': Ag_param},
                            open(data_path + '/result/K_CILM/Ag_param' + str(task_id) + '.pkl', 'wb'),
                            pickle.HIGHEST_PROTOCOL)

                optimizer_CNN.step()
                optimizer_csc.step()

                pred = output.cpu().detach().numpy()
                target = labels.cpu().detach().numpy()
                target[target >= 2] = -1
                acc, acc_P, acc_N, mauc, mmcc, muf1, muar = evaluation_metrics.compute_tagbias_acc_s_m(pred, target)
                mAP, OP, OR, OF1, CP, CR, CF1 = evaluation_metrics.calculate_metrics(pred, target)
                print(
                    'epoch, mean Train: acc, acc_P, acc_N, mauc, mmcc, muf1, muar, mAP, OP, OR, OF1, CP, CR, CF1: %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f,%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n' % (
                        epoch, acc, acc_P, acc_N, mauc, mmcc, muf1, muar, mAP, OP, OR, OF1, CP, CR, CF1))

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes
        print('{} Loss: {:.4f}'.format(
            'train', epoch_loss))
        scheduler_CNN.step(np.mean(epoch_loss))
        scheduler_csc.step(np.mean(epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return CNN, csc


def test_model(CNN, csc, optimizer_CNN, optimizer_csc, class_token,
               Ag_param, task_id,
               label_num):

    since = time.time()
    CNN.eval()
    csc.eval()
    for index, (image_ids, inputs, labels) in enumerate(test_loader):
        # if index > 0:
        #     break

        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        labels = labels[:, 0:label_num]
        sizeD = 28
        inputs = inputs.permute(0, 2, 1, 3)
        input = torch.unsqueeze(inputs, dim=1)
        inputs = torch.cat((input[:, :, 0:sizeD, :, :], input[:, :, sizeD:2 * sizeD, :, :],
                            input[:, :, 2 * sizeD:3 * sizeD, :, :],
                            input[:, :, 3 * sizeD:4 * sizeD, :, :]), dim=1)

        optimizer_CNN.zero_grad()
        optimizer_csc.zero_grad()

        with torch.set_grad_enabled(False):
            feature = CNN(inputs)
            F = feature["attentions"][3]
            output = csc(class_token, F, Ag_param)
            outputs = output[:, 0:labels.size(1)]
            if index == 0:
                outputs_test = outputs
                labels_test = labels
            else:
                outputs_test = torch.cat((outputs_test, outputs), 0)
                labels_test = torch.cat((labels_test, labels), 0)

    pred = outputs_test.to(torch.device("cpu")).numpy()
    target = labels_test.to(torch.device("cpu")).numpy()
    target[target >= 2] = -1

    acc, acc_P, acc_N, mauc, mmcc, muf1, muar = evaluation_metrics.compute_tagbias_acc_s_m(pred, target)
    mAP, OP, OR, OF1, CP, CR, CF1 = evaluation_metrics.calculate_metrics(pred, target)
    print(
        'mean Test: acc, acc_P, acc_N, mauc, mmcc, muf1, muar, mAP, OP, OR, OF1, CP, CR, CF1: %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n' % (
            target.shape[1], acc, acc_P, acc_N, mauc, mmcc, muf1, muar, mAP, OP, OR, OF1, CP, CR, CF1))

    return mAP, OP, OR, OF1, CP, CR, CF1


data_path = '/data1/DataSet/gbm_data/pipeline_data_usecaptk/fengyy/CMU_BrainTumorGene/CL_gene_data/'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    MAP = 0
    OP = 0
    OR = 0
    OF1 = 0
    CP = 0
    CR = 0
    CF1 = 0

    split_type = 'B0_C4'
    num_tasks = 3
    num_classes = 4

    # split_type = 'B0_C3'
    # num_tasks = 4
    # num_classes = 3

    num_epochs = 30
    bs = 4
    init_lr = 0.0001
    print('###batch_size###', bs)
    t = 0.4
    print('t1', t)
    print('t2', 0.3)
    test_results = dict()

    for task_id in range(num_tasks):
        task_id = str(task_id)
        print('#################task' + task_id + '##################:')
        train_datasets = load3DDataset(data_path + '/task/'+split_type+'/train' + str(int(task_id) + 1) + '.csv',
                                       data_path + '/data', data_transforms['train'])
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=bs, shuffle=True, num_workers=4)
        dataset_sizes = len(train_datasets)

        if (int(task_id) == 0):
            # 初始化Ag1，并保存
            class_token = torch.nn.init.kaiming_normal_(torch.empty(num_classes, 2048), mode='fan_out',
                                                        nonlinearity='relu').to(device)
            pickle.dump({'class_token': class_token},
                        open(data_path + '/result/K_CILM/class_token' + str(task_id) + '.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)

            Ag_param = torch.nn.init.uniform_(torch.empty(1, 3), 0, 1).to(device)

            pickle.dump({'Ag_param': Ag_param},
                        open(data_path + '/result/K_CILM/Ag_param' + str(task_id) + '.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)

            CNN = resnet101(pretrained=True, sample_input_D=28, sample_input_H=224, sample_input_W=224,
                            num_seg_classes=num_classes)
            csc = cscBlock(device, 2048, 2048, 2048)
            pre_CNN = None
            pre_csc = None
            pre_Ag_param = None
            pre_class_token = None
        else:
            pre_CNN = copy.deepcopy(CNN)
            pre_csc = copy.deepcopy(csc)
            result = pickle.load(open(data_path + '/result/K_CILM/class_token' + str(int(task_id) - 1) + '.pkl', 'rb'))
            pre_class_token = result['class_token'].to(device)
            class_token = torch.nn.init.kaiming_normal_(
                torch.empty((int(task_id) + 1) * num_classes, 2048), mode='fan_out',
                nonlinearity='relu').to(device)
            class_token[:int(task_id) * num_classes, :] = pre_class_token


            result = pickle.load(open(data_path + '/result/K_CILM/Ag_param' + str(int(task_id) - 1) + '.pkl', 'rb'))
            pre_Ag_param = result['Ag_param'].to(device)
            Ag_param = result['Ag_param'].to(device)
            print('Ag_param[0]: %f, Ag_param[1]: %f Ag_param[2]: %f\n' % (Ag_param[0, 0], Ag_param[0, 1], Ag_param[0, 2]))

        optimizer_CNN = optim.AdamW(CNN.parameters(), lr=init_lr)
        optimizer_csc = optim.AdamW(csc.parameters(), lr=init_lr)
        CNN = CNN.to(device)
        csc = csc.to(device)

        criterion = BinaryCrossEntropyLoss_aw()
        dist_criterion = nn.MultiLabelSoftMarginLoss()
        maxEntropy_criterion = CrossEntropyLoss_aw()

        CNN, csc = train_model(CNN, csc, Ag_param, class_token, criterion,
                                        dist_criterion,
                                        maxEntropy_criterion,
                                        optimizer_CNN, optimizer_csc,
                                        int(task_id),
                                        (int(task_id) + 1) * num_classes,
                                        pre_Ag_param,
                                        pre_CNN=pre_CNN, pre_csc=pre_csc,
                                        pre_class_token=pre_class_token,
                                        num_epochs=num_epochs, init_lr=init_lr)
        torch.save(CNN.state_dict(), data_path + '/result/K_CILM/CNN_model' + str(task_id) + '.pt')
        torch.save(csc.state_dict(), data_path + '/result/K_CILM/csc_model' + str(task_id) + '.pt')

        ###############################################################################################################

        print('task' + task_id + 'performance:')

        test_results[int(task_id)] = []
        for i in range((int(task_id) + 1)):
            test_datasets = load3DDataset(data_path + '/task/'+split_type+'/test' + str(i + 1) + '.csv',
                                          data_path + '/data', data_transforms['test'])
            test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=bs, shuffle=False)

            acc, acc_P, acc_N, mauc, mmcc, muf1, muar = test_model(CNN, csc, optimizer_CNN,
                                                       optimizer_csc, class_token,
                                                       Ag_param, i,
                                                       (i + 1) * num_classes)
            test_results[int(task_id)].append([acc, acc_P, acc_N, mauc, mmcc, muf1, muar])

        print("test_results", test_results)
        MAP_TASK_AVERAGE = 0
        for i in range((int(task_id) + 1)):
            MAP_TASK_AVERAGE += test_results[int(task_id)][i][6]
        MAP_TASK_AVERAGE /= (int(task_id) + 1)
        print("Task ", int(task_id))
        print("Map_AVERAGE", MAP_TASK_AVERAGE)

    forget_list_7 = []
    for i in range(num_tasks - 1):
        forget_list_7.append([(test_results[i][i][0] - test_results[num_tasks - 1][i][0]) / test_results[i][i][0],
                              (test_results[i][i][1] - test_results[num_tasks - 1][i][1]) / test_results[i][i][1],
                              (test_results[i][i][2] - test_results[num_tasks - 1][i][2]) / test_results[i][i][2],
                              (test_results[i][i][3] - test_results[num_tasks - 1][i][3]) / test_results[i][i][3],
                              (test_results[i][i][4] - test_results[num_tasks - 1][i][4]) / test_results[i][i][4],
                              (test_results[i][i][5] - test_results[num_tasks - 1][i][5]) / test_results[i][i][5],
                              (test_results[i][i][6] - test_results[num_tasks - 1][i][6]) / test_results[i][i][6]])
        print("forget_list_7", forget_list_7[i])

    forget_end = []
    for i in range(7):
        forget_end.append((forget_list_7[0][i] + forget_list_7[1][i]) / (num_tasks - 1))

    forget_list_7_4 = []
    for i in range(7):
        forget_list_7_4.append(
            (test_results[num_tasks - 1][0][i] + test_results[num_tasks - 1][1][i] + test_results[num_tasks - 1][2][
                i]) / num_tasks)

    print("forget muar:", forget_end[6])
    print("forget mauc:", forget_end[3])
    print("forget mmcc:", forget_end[4])
    print("forget muf1:", forget_end[5])
    print("forget acc:", forget_end[0])
    print("forget acc_P:", forget_end[1])
    print("forget acc_N:", forget_end[2])

    print('\n')

    print("muar:", forget_list_7_4[6])
    print("mauc:", forget_list_7_4[3])
    print("mmcc:", forget_list_7_4[4])
    print("muf1:", forget_list_7_4[5])
    print("acc:", forget_list_7_4[0])
    print("acc_P:", forget_list_7_4[1])
    print("acc_N:", forget_list_7_4[2])

