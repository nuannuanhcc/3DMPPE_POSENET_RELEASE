import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from dataset import DatasetLoader
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_pose_net
from utils.gcn_utils import adj_mx_from_skeleton
# dynamic dataset import
for i in range(len(cfg.trainset)):
    exec('from ' + cfg.trainset[i] + ' import ' + cfg.trainset[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])

        return start_epoch, model, optimizer

class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_loader = []
        batch_generator = []
        iterator = []
        for i in range(len(cfg.trainset)):
            if i > 0:
                ref_joints_name = trainset_loader[0].joints_name  # 多个数据集时只使用第一个的joint吗？？
            else:
                ref_joints_name = None
            trainset_loader.append(DatasetLoader(eval(cfg.trainset[i])("train"), ref_joints_name, True, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                                                        )))
            batch_generator.append(DataLoader(dataset=trainset_loader[-1], batch_size=cfg.num_gpus*cfg.batch_size//len(cfg.trainset), shuffle=True, num_workers=cfg.num_thread, pin_memory=True))
            iterator.append(iter(batch_generator[-1]))
        
        self.joint_num = trainset_loader[0].joint_num
        self.skeleton = trainset_loader[0].skeleton
        self.itr_per_epoch = math.ceil(trainset_loader[0].__len__() / cfg.num_gpus / (cfg.batch_size // len(cfg.trainset)))
        self.batch_generator = batch_generator
        self.iterator = iterator
        # use skeleton construct graph
        self.adj_mx = adj_mx_from_skeleton(self.joint_num, self.skeleton)
    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_pose_net(cfg, True, self.joint_num, self.adj_mx)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(cfg.testset)("test")
        testset_loader = DatasetLoader(testset, None, False, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                                                        ))
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.batch_generator = batch_generator
        self.adj_mx = adj_mx_from_skeleton(self.joint_num, self.skeleton)

    def _make_model(self):
        
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(cfg, False, self.joint_num, self.adj_mx)
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model

    def _evaluate(self, preds, result_save_path, global_steps=None):
        self.testset.evaluate(preds, result_save_path, global_steps)

