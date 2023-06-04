import os
import time
from os import path as osp

import numpy
import numpy as np
import torch
import json

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_glob_speed import *
from transformations import *
from model_resnet1d import *

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}


def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(imu_sequence, args, **kwargs):
    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    dataset = StridedSequenceDataset(
        imu_sequence, seq_type, args.step_size, args.window_size,
        random_shift=0, transform=None,
        shuffle=False, grv_only=True, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def recon_traj_with_preds(dataset, prev_pos, preds, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts
    dts = np.mean(ts[-1] - ts[0])
    pos = np.array(preds) * dts + prev_pos
    return pos


def test_sequence(args):

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []

    prev_pos = np.zeros((1,2))
    all_pos_pred = np.array(prev_pos)
    if True:
        print("Started to process IMU Data.......")
        # give imu_data (200 x 7) tensor [[Ax,Ay,Az, Wx,Wy,Wz, t ],[],[],.....] as imu_sequence
        # str_input = input("enter input:")
        with open('input.txt', 'r') as file:
            # Read the contents of the file
            str_input = file.read()

        imu_sequence = np.array(eval(str_input))

        network = get_model(args.arch)

        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval().to(device)
        print('Model {} loaded to device {}.'.format(args.model_path, device))

        seq_dataset = get_dataset(imu_sequence, args)
        seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False)
        pred = None
        for bid, (feat, _, _) in enumerate(seq_loader):
            pred = network(feat.to(device)).cpu().detach().numpy()

        print("Predicted velocity ", pred)
        preds_seq.append(pred)
        pos_pred = recon_traj_with_preds(seq_dataset,prev_pos, pred)[:, :2]
        print("Predicted position ", pos_pred)
        all_pos_pred = np.concatenate((all_pos_pred, pos_pred))

        # Plot figures
        kp = pred.shape[1]

        plt.figure('{}'.format("Output Trajectory"), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(all_pos_pred[:, 0], all_pos_pred[:, 1])
        plt.title("Output Trajectory")
        plt.axis('equal')
        plt.legend(['Predicted'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.tight_layout()

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, "Output Trajectory" + '_gsn.npy'),
                    [pos_pred[:, :2]])
            plt.savefig(osp.join(args.out_dir, "Output Trajectory" + '_gsn.png'))

        plt.close('all')
        print('---------------------------------------------------------')


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'test':
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')