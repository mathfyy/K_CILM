import os
import time
import numpy as np
import torch
import pandas as pd
import K_CILM.evaluation_metrics as evaluation_metrics
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from torchvision import transforms
import pickle
import torch.nn as nn

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
        return image_id[0], image, label, self.labels.iloc[idx, 0]


def eval_model(outputs_test, labels_test, names_test):
    pred = outputs_test.to(torch.device("cpu")).numpy()
    target = labels_test.to(torch.device("cpu")).numpy()

    # 保存预测结果
    preds = (pred > 0.5).astype(np.int64)
    tensor_dict = {'names': list(names_test)}
    for i in range(preds.shape[1]):
        new_key = f'preds{i}'
        tensor_dict[new_key] = torch.tensor(preds[:, i])
    # df = pd.DataFrame(tensor_dict)
    # df.to_csv(model_path + '/pred_task' + str(task_id) + '.csv', index=False)
    acc, acc_P, acc_N, mauc, mmcc, muf1, muar = evaluation_metrics.compute_tagbias_acc_s_m(pred, target)
    mAP, OP, OR, OF1, CP, CR, CF1 = evaluation_metrics.calculate_metrics(pred, target)
    print(
        'mean Test: mAP, CP, CR, CF1, OP, OR, OF1, muf1, muar, mauc, acc_P, acc_N, acc, diff, pn, mmcc: %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n' % (
            target.shape[1], mAP, CP, CR, CF1, OP, OR, OF1, muf1, muar, mauc, acc_P, acc_N, acc, abs(acc_P - acc_N),
            (acc_P + acc_N) / 2.0, mmcc))

    return mAP, OP, OR, OF1, CP, CR, CF1


def test_model_K_CILM(CNN, csc, class_token, Ag_param, task_id, label_num):
    for index, (image_ids, inputs, labels, names) in enumerate(test_loader):
        if index > 2:
            break

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

        with torch.set_grad_enabled(False):
            feature = CNN(inputs)
            F = feature["attentions"][3]
            output = csc(class_token, F, Ag_param)
            outputs = output[:, 0:labels.size(1)]
            if index == 0:
                outputs_test = outputs
                labels_test = labels
                names_test = names
            else:
                outputs_test = torch.cat((outputs_test, outputs), 0)
                labels_test = torch.cat((labels_test, labels), 0)
                names_test = names_test + names

    mAP, OP, OR, OF1, CP, CR, CF1 = eval_model(outputs_test, labels_test, names_test)
    return mAP, OP, OR, OF1, CP, CR, CF1


from K_CILM.resnet3D import resnet101, cscBlock

if __name__ == '__main__':

    split_type = 'B0_C4'
    num_tasks = 3
    num_classes = 4

    # split_type = 'B0_C3'
    # num_tasks = 4
    # num_classes = 3

    model_type = 'K_CILM'
    data_path = '/data1/DataSet/gbm_data/pipeline_data_usecaptk/fengyy/CMU_BrainTumorGene/CL_gene_data/'
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model_path = data_path + '/result/' + split_type + '/' + model_type + '/'

    bs = 4
    print('###batch_size###', bs)
    t = 0.4
    print('t1', t)
    print('t2', 0.3)
    test_results = dict()
    acc = 0.0
    acc_P = 0.0
    acc_N = 0.0
    mauc = 0.0
    mmcc = 0.0
    muf1 = 0.0
    muar = 0.0

    for task_id in range(num_tasks):
        task_id = str(task_id)
        print('task' + task_id + 'performance:')

        test_results[int(task_id)] = []
        for i in range((int(task_id) + 1)):
            test_datasets = load3DDataset(data_path + '/task/' + split_type + '/test' + str(i + 1) + '.csv',
                                          data_path + '/data', data_transforms['test'])
            test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=bs, shuffle=False)

            CNN = resnet101(pretrained=True, sample_input_D=28, sample_input_H=224, sample_input_W=224,
                            num_seg_classes=num_classes).to(device)
            csc = cscBlock(device, 2048, 2048, 2048).to(device)
            CNN.load_state_dict(torch.load(model_path + '/CNN_model' + str(task_id) + '.pt'))
            csc.load_state_dict(torch.load(model_path + '/csc_model' + str(task_id) + '.pt'))

            # param
            result = pickle.load(open(model_path + '/class_token' + str(int(task_id)) + '.pkl', 'rb'))
            class_token = result['class_token'].to(device)
            result = pickle.load(open(model_path + '/Ag_param' + str(int(task_id)) + '.pkl', 'rb'))
            Ag_param = result['Ag_param'].to(device)

            acc, acc_P, acc_N, mauc, mmcc, muf1, muar = test_model_K_CILM(CNN, csc, class_token, Ag_param, i,
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
