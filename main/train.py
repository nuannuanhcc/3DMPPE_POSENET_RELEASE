import argparse
from config import cfg
import torch
from base import Trainer, Tester
import torch.backends.cudnn as cudnn
import neptune
from utils.pose_utils import flip
import numpy as np
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--model_name', type=str, dest='model_name')
    parser.add_argument('--neptune', dest='neptune_use', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def valid(trainer, valider, global_steps):
    preds = []
    trainer.model.eval()
    with torch.no_grad():
        for itr, input_img in enumerate(tqdm(valider.batch_generator)):

            # forward
            coord_out = trainer.model(input_img)

            if cfg.flip_test:
                flipped_input_img = flip(input_img, dims=3)
                flipped_coord_out = trainer.model(flipped_input_img)
                flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
                for pair in valider.flip_pairs:
                    flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:,
                                                                                         pair[1],
                                                                                         :].clone(), flipped_coord_out[
                                                                                                     :, pair[0],
                                                                                                     :].clone()
                coord_out = (coord_out + flipped_coord_out) / 2.

            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)

    # evaluate
    preds = np.concatenate(preds, axis=0)
    valider._evaluate(preds, cfg.result_dir, global_steps)

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.model_name, args.gpu_ids, args.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    valider = Tester(100)
    valider._make_batch_generator()

    # neptune
    global_steps = None
    if args.neptune_use:
        neptune.init('hccccccccc/3DMPPE-POSENET')
        neptune.create_experiment(args.model_name)
        neptune.append_tag('pose')
        global_steps = {
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }


    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.model.train()
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr in range(trainer.itr_per_epoch):
            
            input_img_list, joint_img_list, joint_vis_list, joints_have_depth_list = [], [], [], []
            for i in range(len(cfg.trainset)):
                try:
                    input_img, joint_img, joint_vis, joints_have_depth = next(trainer.iterator[i])
                except StopIteration:
                    trainer.iterator[i] = iter(trainer.batch_generator[i])
                    input_img, joint_img, joint_vis, joints_have_depth = next(trainer.iterator[i])

                input_img_list.append(input_img)
                joint_img_list.append(joint_img)
                joint_vis_list.append(joint_vis)
                joints_have_depth_list.append(joints_have_depth)
            
            # aggregate items from different datasets into one single batch
            input_img = torch.cat(input_img_list,dim=0)
            joint_img = torch.cat(joint_img_list,dim=0)
            joint_vis = torch.cat(joint_vis_list,dim=0)
            joints_have_depth = torch.cat(joints_have_depth_list,dim=0)
            
            # shuffle items from different datasets
            rand_idx = []
            for i in range(len(cfg.trainset)):
                rand_idx.append(torch.arange(i,input_img.shape[0],len(cfg.trainset)))
            rand_idx = torch.cat(rand_idx,dim=0)
            rand_idx = rand_idx[torch.randperm(input_img.shape[0])]
            input_img = input_img[rand_idx]; joint_img = joint_img[rand_idx]; joint_vis = joint_vis[rand_idx]; joints_have_depth = joints_have_depth[rand_idx];
            target = {'coord': joint_img, 'vis': joint_vis, 'have_depth': joints_have_depth}

            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()
            
            # forward
            loss_coord, loss_all, log_var = trainer.model(input_img, target)
            loss_coord = loss_coord.mean()
            loss_all = loss_all.mean()
            var = torch.exp(log_var.reshape(-1) / 2)
            var = var.mean()


            # backward
            loss = loss_all
            loss.backward()
            trainer.optimizer.step()
            
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f' % ('loss_coord', loss_coord.detach()),
                '%s: %.4f' % ('loss_all', loss_all.detach()),
                '%s: %.4f' % ('var', var.detach()),
                ]
            trainer.logger.info(' '.join(screen))
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            if args.neptune_use and itr % 50 == 0:
                neptune_step = global_steps['train_global_steps']
                neptune.send_metric('batch_loss', neptune_step, loss_coord.cpu().detach().numpy())
                neptune.send_metric('batch_loss_all', neptune_step, loss_all.cpu().detach().numpy())
                neptune.send_metric('var', neptune_step, var.cpu().detach().numpy())
                neptune.send_metric('lr', neptune_step, trainer.get_lr())
                global_steps['train_global_steps'] = neptune_step + 1

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

        valid(trainer, valider, global_steps)
    neptune.stop()
    
if __name__ == "__main__":
    main()
