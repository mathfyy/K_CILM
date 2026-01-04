import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

data_path = '/data1/DataSet/gbm_data/pipeline_data_usecaptk/fengyy/CMU_BrainTumorGene/CL_gene_data/pretrain_model/'

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block,
#                  layers,
#                  sample_input_D,
#                  sample_input_H,
#                  sample_input_W,
#                  num_seg_classes,
#                  shortcut_type='B',
#                  no_cuda=False):
#         self.inplanes = 64
#         self.no_cuda = no_cuda
#         super(ResNet, self).__init__()
#
#         self.up = nn.Upsample(scale_factor=(2, 1, 1))
#         self.norm = nn.InstanceNorm3d(4, eps=1e-5, affine=False)
#         # self.norm = nn.InstanceNorm3d(1, eps=1e-5, affine=True)
#
#         self.conv1 = nn.Conv3d(
#             4,
#             64,
#             kernel_size=7,
#             stride=(2, 2, 2),
#             padding=(3, 3, 3),
#             bias=False)
#
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(1, 1, 1))
#         self.layer2 = self._make_layer(
#             block, 128, layers[1], shortcut_type, stride=(2, 2, 2))
#         self.layer3 = self._make_layer(
#             block, 256, layers[2], shortcut_type, stride=(1, 1, 1), dilation=2)
#         self.layer4 = self._make_layer(
#             block, 512, layers[3], shortcut_type, stride=(1, 1, 1), dilation=4)
#
#         # 最佳 avg 64%
#         self.maxpool2 = nn.MaxPool3d((7, 28, 28), (7, 28, 28))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), dilation=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride,
#                     no_cuda=self.no_cuda)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.up(x)
#         x = self.norm(x)
#         x = self.relu(self.bn1(self.conv1(x)))
#
#         x = self.maxpool(x)
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         # embeddings = self.conv_seg(x4)
#
#         embeddings = self.maxpool2(x4)  # 8 * 2048 * 1 * 1
#         embeddings = embeddings.view(embeddings.size(0), -1)  # 8 * 2048
#         # logits = self.fc1(x)
#
#         outputs = {
#             "embeddings": embeddings,
#             "attentions": [x1, x2, x3, x4]
#         }
#
#         return outputs
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()

        self.norm = nn.InstanceNorm3d(1, eps=1e-5, affine=False)

        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(2, 2, 2))
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=(2, 2, 2))
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=(1, 1, 1), dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=(1, 1, 1), dilation=4)

        # 最佳 avg 64%
        self.maxpool2 = nn.MaxPool3d((7, 14, 14), (7, 14, 14))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = torch.cat((x[:, 0, :, :, :], x[:, 1, :, :, :],
                       x[:, 2, :, :, :], x[:, 3, :, :, :]), dim=1).unsqueeze(1)

        x = self.norm(x)
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # embeddings = self.conv_seg(x4)

        outputs = {
            # "embeddings": embeddings,
            "attentions": [x1, x2, x3, x4]
        }

        return outputs


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.device = device
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('GCN is running')
        # print(input.shape)
        # input = torch.Tensor(input)
        # input = input.to(self.device)
        support = torch.matmul(input, self.weight)
        # support = support.to(self.device)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class labelCorModule_s(nn.Module):
    def __init__(self, ):
        super(labelCorModule_s, self).__init__()
        self.tanh = nn.Tanh()
        self.upCor_s = upCor_s()
        self.downCor_s = downCor_s()

    def forward(self, Ag, V0, M, Ag_param):
        # x0 = Ag
        cor_D = torch.bmm(V0, V0.permute(0, 2, 1)) / V0.shape[2]
        cor_P = torch.bmm(M.permute(0, 2, 1), M) / M.shape[1]
        # x0 = Ag + torch.mean(
        #     Ag_param[0, 0] * self.tanh(self.bn(cor_D.unsqueeze(1)).squeeze(1)) + Ag_param[0, 1] * self.tanh(
        #         self.bn(cor_P.unsqueeze(1)).squeeze(1)), dim=0)
        x0 = Ag + torch.mean(
            Ag_param[0, 0] * (self.tanh(cor_D.unsqueeze(1)).squeeze(1)) + Ag_param[0, 1] * (
                self.tanh(cor_P.unsqueeze(1)).squeeze(1)), dim=0)
        x0 = x0.unsqueeze(0).unsqueeze(0)

        # 0
        # out = x0

        # 1
        # P, Q, Atte = self.upCor_s(x0)
        # out = self.downCor_s(x0, P, Q)
        # out = Ag_param[0, 2] * out + x0

        # 2
        P, Q, Atte = self.upCor_s(x0, True)
        P1, Q1, Atte1 = self.upCor_s(Atte.unsqueeze(1))
        out = self.downCor_s(x0, P, self.downCor_s(Q.unsqueeze(1), P1, Q1))
        out = Ag_param[0, 2] * out + x0

        # 3
        # P, Q, Atte = self.upCor_s(x0, True)
        # P1, Q1, Atte1 = self.upCor_s(Atte.unsqueeze(1), True)
        # P2, Q2, Atte2 = self.upCor_s(Atte1.unsqueeze(1))
        # dQ2 = self.downCor_s(Q1.unsqueeze(1), P2, Q2)
        # dQ1=self.downCor_s(Q.unsqueeze(1), P1, dQ2)
        # out = self.downCor_s(x0, P, dQ1)
        # out = Ag_param[0, 2] * out + x0
        return out


class upCor_s(nn.Module):
    def __init__(self, ):
        super(upCor_s, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        in_dim = 1
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x, is_atte=False):
        m_batchsize, C, height, width = x.size()
        P = self.sigmoid(self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1))
        Q = self.sigmoid(self.key_conv(x).view(m_batchsize, -1, width * height))
        Atte = []
        if is_atte is True:
            Atte = torch.bmm(P, Q) / (width * height)
        return P, Q, Atte


class downCor_s(nn.Module):
    def __init__(self, ):
        super(downCor_s, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, P, Q):
        m_batchsize, C, height, width = x.size()
        x = x.view(m_batchsize, -1, width * height)
        P = P.view(m_batchsize, width * height, -1)
        Q = Q.view(m_batchsize, -1, width * height)
        out1 = torch.bmm(x, P) / (width * height)
        out1 = self.tanh(out1)
        out = torch.bmm(out1, Q) / (width * height)
        out = out.view(m_batchsize, height, width)
        out = self.tanh(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // heads  # 每个头的维度
        self.h = heads  # 头的数量

        # 以下三个是线性层，用于处理Q（Query），K（Key），V（Value）
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)  # Dropout层
        # self.out = nn.Linear(d_model, d_model)  # 输出层

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        # torch.matmul是矩阵乘法，用于计算query和key的相似度
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # 在第一个维度增加维度
            scores = scores.masked_fill(mask == 0, -1e9)  # 使用mask将不需要关注的位置设置为一个非常小的数

        # 对最后一个维度进行softmax运算，得到权重
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)  # 应用dropout

        output = torch.matmul(scores, v)  # 将权重应用到value上
        return scores, output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)  # 获取batch_size

        # 将Q, K, V通过线性层处理，然后分割成多个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 转置来获取维度为bs * h * sl * d_model的张量
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 调用attention函数计算输出
        scores, output = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # 重新调整张量的形状，并通过最后一个线性层
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # output = self.out(concat)  # 最终输出
        return scores, concat


class cadBlock(nn.Module):
    def __init__(self, device, mid_dim=2048, head_dim=512):
        super(cadBlock, self).__init__()
        self.device = device
        self.ln_K = nn.LayerNorm(mid_dim)
        self.ln_V = nn.LayerNorm(mid_dim)
        self.ln_Q = nn.LayerNorm(mid_dim)

        self.mca = MultiHeadAttention(mid_dim, head_dim)

        self.mlp = nn.Linear(mid_dim, 1)

    def forward(self, F_data, A_cls):
        F_data = F_data.reshape(F_data.shape[0], F_data.shape[1],
                                F_data.shape[2] * F_data.shape[3] * F_data.shape[4]).transpose(2, 1)
        Q_l = self.ln_Q(A_cls)
        K_l = self.ln_K(F_data)
        V_l = self.ln_V(F_data)

        scores, Hba = self.mca(Q_l.repeat(K_l.shape[0], 1, 1), K_l, V_l)
        scores = scores.permute(0, 1, 3, 2).reshape(scores.shape[0], -1, scores.shape[2])
        out = self.mlp(Hba).reshape(Hba.shape[0], Hba.shape[1])

        return scores, Hba, out


class cscBlock(nn.Module):
    def __init__(self, device, D0, D1, D2, mid_dim=2048, head_dim=512):
        super(cscBlock, self).__init__()
        self.device = device
        self.D0 = D0
        self.D1 = D1
        self.D2 = D2
        self.mid_dim = mid_dim
        self.head_dim = head_dim
        # self.gp = nn.AdaptiveAvgPool2d((1, 1))

        self.cad = cadBlock(device)

        self.generalGCN = GraphConvolution(D0, D1)
        # self.specificGCN = GraphConvolution(2 * D1, D2)
        self.lRelu = nn.LeakyReLU(inplace=True)

        # self.agp = nn.AdaptiveAvgPool2d((1, D1))
        # self.conv_ex = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn_ex = nn.BatchNorm2d(1)
        # self.relu_ex = nn.LeakyReLU(inplace=True)
        #
        # self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(1)
        # self.relu = nn.LeakyReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

        # self.labelCorModule = labelCorModule()
        self.labelCorModule = labelCorModule_s()

        self.fc = nn.Linear(D2, 1)

    #     self.initialize_parameters()
    #
    # def initialize_parameters(self):
    #     proj_std = (self.mid_dim ** -0.5) * (2 ** -0.5)
    #     fc_std = (2 * self.mid_dim) ** -0.5
    #
    #     nn.init.normal_(self.cad.ln_K.weight, std=fc_std)
    #     nn.init.normal_(self.cad.ln_V.weight, std=fc_std)
    #     nn.init.normal_(self.cad.ln_Q.weight, std=fc_std)
    #
    #     nn.init.normal_(self.cad.mca.k_linear.weight, std=proj_std)
    #     nn.init.normal_(self.cad.mca.q_linear.weight, std=proj_std)
    #     nn.init.normal_(self.cad.mca.v_linear.weight, std=proj_std)
    #     nn.init.normal_(self.cad.mlp.weight, std=fc_std)
    #
    #     nn.init.normal_(self.fc.weight, std=fc_std)
    #
    #     nn.init.normal_(self.labelCorModule.upCor_s.key_conv.weight, std=fc_std)
    #     nn.init.normal_(self.labelCorModule.upCor_s.query_conv.weight, std=fc_std)

    def forward(self, class_token, F, Ag_param):
        Ag = torch.tanh(torch.mm(class_token, class_token.permute(1, 0)) / class_token.shape[1])

        scores, V0, output1 = self.cad(F, class_token)

        # 基于影像特征和标签关注的位置，来优化关联矩阵
        # 避免对样本分布依赖过重
        Ag_ = self.labelCorModule(Ag, V0, scores, Ag_param)

        V1 = torch.ones(size=(V0.size(0), V0.size(1), self.D1)).to(self.device)
        for i in range(V0.size(0)):
            V1[i,] = self.lRelu(self.generalGCN(V0[i, :, :], Ag_))

        # V = self.relu_ex(self.bn_ex(self.conv_ex(self.agp(V1.unsqueeze(dim=1)))))
        # V = V.repeat(1, 1, V1.size(1), 1).squeeze(dim=1)
        # V1_ = torch.cat((V1, V), dim=2)
        #
        # W = self.relu(self.bn(self.conv(V1_.unsqueeze(dim=1)))).squeeze(dim=1)
        # As_ = self.sigmoid(
        #     torch.bmm(V1_.view(V1_.size(0), V1_.size(1), -1), W.view(W.size(0), W.size(1), -1).permute(0, 2, 1)) /
        #     W.shape[2])
        # As = torch.mean(As_, dim=0)
        #
        # V2 = torch.ones(size=(V1.size(0), V1.size(1), self.D2)).to(self.device)
        # for i in range(V1.size(0)):
        #     V2[i,] = self.lRelu(self.specificGCN(V1_[i, :, :], As))

        output2 = self.fc(V1).squeeze(dim=2)
        output = self.sigmoid(output1 + output2)

        return output


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(data_path + '/resnet_50.pth'), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        '''
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        '''
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),strict=False)
        # cocodata
        # model.load_state_dict(torch.load(data_path + '/resnet101-5d3b4d8f.pth'), strict=False)
        # chestdata
        model.load_state_dict(torch.load(data_path + '/resnet_101.pth'), strict=False)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
