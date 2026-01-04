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

from resnet2D import resnet101, cscBlock

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


class loadPicDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file, header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1:].values
        image_id = self.labels.iloc[idx, :].values
        label = label.astype('double')
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
            image = np.concatenate((image, image, image), axis=2)
        if self.transform:
            image = self.transform(image)
        # id = torch.Tensor(image_id[0])
        return image_id[0], image, label


# trainval需要水平翻转，val与test不需要水平翻转
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.488, 0.486, 0.406], [0.229, 0.224, 0.228])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.488, 0.486, 0.406], [0.229, 0.224, 0.228])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.488, 0.486, 0.406], [0.229, 0.224, 0.228])
    ])
}


def train_model(CNN, csc, Ag_param, class_token, criterion, distill_criterion,
                maxEntropy_criterion, optimizer_CNN,
                optimizer_csc,
                task_id, label_num, pre_Ag_param, pre_CNN=None, pre_classifier=None, pre_csc=None,
                pre_class_token=None,
                num_epochs=1, init_lr=0.001):
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
                    lambdas = 0.75
                    for ic in range(pre_output.shape[1]):
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
                # 获取更新后的
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


data_path = '/data1/DataSet/gbm_data/pipeline_data_usecaptk/fengyy/chestmnist/'
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

    num_epochs = 20
    bs = 48
    init_lr = 0.0001
    print('###batch_size###', bs)
    t = 0.4
    print('t1', t)
    print('t2', 0.3)
    test_results = dict()

    for task_id in range(num_tasks):
        task_id = str(task_id)
        print('#################task' + task_id + '##################:')
        train_datasets = loadPicDataset(data_path + '/task/' + split_type + '/train' + str(int(task_id) + 1) + '.csv',
                                        data_path + '/data', data_transforms['train'])

        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=bs, shuffle=True, num_workers=4)
        dataset_sizes = len(train_datasets)

        if (int(task_id) == 0):
            class_token = torch.nn.init.kaiming_normal_(torch.empty(num_classes, 2048), mode='fan_out',
                                                        nonlinearity='relu').to(device)
            pickle.dump({'class_token': class_token},
                        open(data_path + '/result/K_CILM/class_token' + str(task_id) + '.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)

            Ag_param = torch.nn.init.uniform_(torch.empty(1, 3), 0, 1).to(device)

            pickle.dump({'Ag_param': Ag_param},
                        open(data_path + '/result/K_CILM/Ag_param' + str(task_id) + '.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)

            CNN = resnet101(pretrained=True, num_classes=num_classes)
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
            print(
                'Ag_param[0]: %f, Ag_param[1]: %f Ag_param[2]: %f\n' % (Ag_param[0, 0], Ag_param[0, 1], Ag_param[0, 2]))

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
            test_datasets = loadPicDataset(data_path + '/task/' + split_type + '/test' + str(i + 1) + '.csv',
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

    # acc, acc_P, acc_N, mauc, mmcc, muf1, muar
    # OP, OR, OF1, CP, CR, CF1, mAP

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

