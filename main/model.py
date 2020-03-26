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


def soft_argmax(heatmaps, joint_num, scale):  # [32, 1152, 64, 64]

    heatmaps = heatmaps.reshape(
        (-1, joint_num, cfg.depth_dim * cfg.output_shape[0] * cfg.output_shape[1]))  # [32, 18, 262144]
    scale = scale.unsqueeze(-1).expand(-1,-1,heatmaps.shape[-1])
    heatmaps = F.softmax(heatmaps * scale, 2)
    heatmaps = heatmaps.reshape(
        (-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))  # [32, 18, 64, 64, 64]

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1, cfg.output_shape[1] + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1, cfg.output_shape[0] + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1, cfg.depth_dim + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1  # ???为什么要减1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


class ResPoseNet(nn.Module):
    def __init__(self, backbone, head, joint_num):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num

        self.log_var_head = nn.Sequential()
        self.log_var_head.add_module('gap', nn.AdaptiveAvgPool2d((1, 1)))
        self.log_var_head.add_module('bottle_neck',
                                     nn.Sequential(
                                         nn.Linear(2048, 256, bias=False),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU(inplace=True),
                                         # nn.Dropout(p=0.5),

                                         nn.Linear(256, 64, bias=False),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(inplace=True),
                                         # nn.Dropout(p=0.5),
                                     ))
        self.log_var_head.add_module('fc', nn.Linear(64, 18))

        nn.init.constant_(self.log_var_head.fc.weight, 0)
        nn.init.constant_(self.log_var_head.fc.bias, 0)

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

    def forward(self, input_img, target=None, s_max=cfg.s_max, s_min=cfg.s_min, thresh=2, lamda=0.1):  # [32, 3, 256, 256]
        print(s_max,s_min)
        fm = self.backbone(input_img)  # [32, 2048, 8, 8]

        x = self.log_var_head.gap(fm[-1])
        x = x.view(*x.shape[:2])
        x = self.log_var_head.bottle_neck(x)
        scale = self.log_var_head.fc(x)
        scale = F.sigmoid(scale)
        scale = (s_max - s_min) * scale + s_min
        hm = self.head(fm[-1])  # [32, 1152, 64, 64]
        coord = soft_argmax(hm, self.joint_num, scale)

        if self.USE_GCN:
            n, k, _ = coord.shape
            coord_norm = torch.empty(n, k, 2).cuda()
            coord_norm[:, :, 0] = coord[:, :, 0] / cfg.output_shape[1] * 2 - 1
            coord_norm[:, :, 1] = coord[:, :, 1] / cfg.output_shape[0] * 2 - 1
            coord_norm = coord_norm.view(n, k, 1, 2)
            feat = []
            for f in fm:
                feat.append(torch.nn.functional.grid_sample(f, coord_norm))
            feat = torch.cat(feat, dim=1)  # [n, 3840, 18, 1]
            feat = feat.view(n, k, -1)  # [n, 18, 3840]

        if target is None:
            if self.USE_GCN:
                coord1 = self.gcn(torch.cat((feat, coord), dim=-1))
                return coord1
            else:
                return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            heatmaps = hm.reshape(
                (-1, self.joint_num, cfg.depth_dim * cfg.output_shape[0] * cfg.output_shape[1]))  # [32, 18, 262144]
            idx = torch.argmax(heatmaps, dim=2, keepdim=True)
            preds = idx.repeat(1, 1, 3)
            preds[:, :, 0] = (preds[:, :, 0]) % cfg.depth_dim
            preds[:, :, 1] = (preds[:, :, 1]) / cfg.depth_dim % cfg.output_shape[1]
            preds[:, :, 2] = (preds[:, :, 2]) / cfg.depth_dim / cfg.output_shape[1]
            dis = torch.abs(preds.type(torch.cuda.FloatTensor) - target_coord) * target_vis
            dis = torch.mean(dis, -1)  # [32, 18]
            mask = dis > thresh

            loss_norm = torch.where(mask, torch.clamp(scale-s_min, min=0), torch.clamp(s_max-scale, min=0))
            loss_norm = loss_norm * target_vis.squeeze(-1)
            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1] + loss_coord[:, :, 2] * target_have_depth) / 3.
            loss = loss_coord + lamda * loss_norm

            if self.USE_GCN:
                coord1 = self.gcn(torch.cat((feat, coord), dim=-1))

                loss_coord1 = torch.abs(coord1 - target_coord) * target_vis
                loss_coord1 = (loss_coord1[:, :, 0] + loss_coord1[:, :, 1] + loss_coord1[:, :,
                                                                             2] * target_have_depth) / 3.
                loss += loss_coord1
                loss_coord += loss_coord1
            return loss_coord.mean(), loss.mean(), scale, loss_norm.mean()


def get_pose_net(cfg, is_train, joint_num):
    backbone = ResNetBackbone(cfg.resnet_type)
    head_net = HeadNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()

    model = ResPoseNet(backbone, head_net, joint_num)
    return model

