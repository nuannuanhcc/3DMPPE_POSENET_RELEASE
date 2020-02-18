import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn
import neptune
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--model_name', type=str, dest='model_name')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.model_name, args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    preds = []

    neptune.init('hccccccccc/sandbox')
    neptune.create_experiment(name='start-with-neptune')

    with torch.no_grad():
        for itr, (input_img, joint_img, joint_vis) in enumerate(tqdm(tester.batch_generator)):
            
            # forward
            coord_out = tester.model(input_img)

            if cfg.flip_test:
                flipped_input_img = flip(input_img, dims=3)
                flipped_coord_out = tester.model(flipped_input_img)
                flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
                for pair in tester.flip_pairs:
                    flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1], :].clone(), flipped_coord_out[:, pair[0], :].clone()
                coord_out = (coord_out + flipped_coord_out)/2.

            vis = False
            if vis:
                # filename = str(itr)
                for idx in range(input_img.shape[0]):
                    tmpimg = input_img[idx].cpu().numpy()
                    tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
                    tmpimg = tmpimg.astype(np.uint8)
                    tmpimg = tmpimg[::-1, :, :]
                    tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
                    tmpkps = np.zeros((3,tester.joint_num))
                    tmpkps[:2,:] = coord_out[idx,:,:2].cpu().numpy().transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
                    tmpkps[2,:] = 1

                    tmpimg_gt = tmpimg.copy()
                    tmpkps_gt = np.zeros((3, tester.joint_num))
                    tmpkps_gt[:2, :] = joint_img[idx, :, :2].cpu().numpy().transpose(1, 0) / cfg.output_shape[0] * cfg.input_shape[0]
                    tmpkps_gt[2, :] = joint_vis[idx].squeeze(-1)

                    tmpimg = vis_keypoints(tmpimg, tmpkps, tester.skeleton)
                    tmpimg_gt = vis_keypoints(tmpimg_gt, tmpkps_gt, tester.skeleton)

                    neptune.log_image('mosaics', tmpimg)
                    neptune.log_image('mosaics', tmpimg_gt)

                    # cv2.imwrite(filename + '_output.jpg', tmpimg)
            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)
            
    # evaluate
    preds = np.concatenate(preds, axis=0)
    tester._evaluate(preds, cfg.result_dir)    
    neptune.stop()
if __name__ == "__main__":
    main()
