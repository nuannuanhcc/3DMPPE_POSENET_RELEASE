import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from config import cfg
from nets.sem_gcn import SemGCN
from utils.gcn_utils import adj_mx_from_skeleton

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

class ResPoseNet(nn.Module):
    def __init__(self, backbone, head, joint_num):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num

        self.USE_GCN = cfg.use_gcn
        if self.USE_GCN:
            if 'Human36M' in cfg.trainset:
                self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                                 (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

            if 'MuCo' in cfg.trainset:
                self.skeleton = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14),
                                 (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
            self.gcn = SemGCN(adj_mx_from_skeleton(self.joint_num, self.skeleton),
                              hid_dim=128, num_layers=4, p_dropout=0.5)

    def forward(self, input_img, target=None): # [32, 3, 256, 256]
        fm = self.backbone(input_img) # [32, 2048, 8, 8]
        hm = self.head(fm) # [32, 1152, 64, 64]
        coord = soft_argmax(hm, self.joint_num)

        if target is None:
            if self.USE_GCN:
                coord1 = self.gcn(coord)
                return coord1
            else:
                return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
            
            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.
            if self.USE_GCN:
                coord1 = self.gcn(coord)
                loss_coord1 = torch.abs(coord1 - target_coord) * target_vis
                loss_coord1 = (loss_coord1[:, :, 0] + loss_coord1[:, :, 1] + loss_coord1[:, :, 2] * target_have_depth) / 3.
                loss_coord += loss_coord1
            return loss_coord

def get_pose_net(cfg, is_train, joint_num):
    
    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = HeadNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()

    model = ResPoseNet(backbone, head_net, joint_num)
    return model

