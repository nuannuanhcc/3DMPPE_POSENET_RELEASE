import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from config import cfg
import math
from torch.nn.parameter import Parameter

class HeadNet(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 256

        super(HeadNet, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

def soft_argmax(heatmaps, joint_num): # [32, 1152, 64, 64]

    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1])) # [32, 18, 262144]
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1])) # [32, 18, 64, 64, 64]

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[1]+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[0]+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1,cfg.depth_dim+1).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) -1 #???为什么要减1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class ResPoseNet(nn.Module):
    def __init__(self, backbone, head, gcn, joint_num):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.gcn = gcn
        self.joint_num = joint_num

    def forward(self, input_img, adj_mx, target=None): # [32, 3, 256, 256]

        fm = self.backbone(input_img) # [32, 2048, 8, 8]
        hm = self.head(fm) # [32, 1152, 64, 64]
        coord = soft_argmax(hm, self.joint_num)

        # if adj_mx:

        coord1 = self.gcn(coord, adj_mx)
        if target is None:
            return coord1
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
            
            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.

            loss_coord1 = torch.abs(coord1 - target_coord) * target_vis
            loss_coord1 = (loss_coord1[:,:,0] + loss_coord1[:,:,1] + loss_coord1[:,:,2] * target_have_depth)/3.
            return loss_coord + loss_coord1

def get_pose_net(cfg, is_train, joint_num):
    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = HeadNet(joint_num)
    gcn = GCN(3, 128, 3)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()
        # gcn.init_weights()
    model = ResPoseNet(backbone, head_net, gcn, joint_num)
    return model

