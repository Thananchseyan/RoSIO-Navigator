import os
import pdb
import time
from os import path as osp

import numpy as np
import torch
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_glob_speed import *
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *
from scipy.spatial.transform import Rotation as R
from torch.nn.functional import normalize

import neptune.new as neptune

# from torchviz import make_dot
# from graphviz import Source
# import pydot
# import torchviz
# import graphviz


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


def targetTransformationModule(input_array, random_degrees, device):
    input_arrayy = input_array.clone()
    theta = math.pi / 4  # angle of rotation in radians
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    random_torch_rotation = []

    # rotation matrix
    Rzz = torch.tensor([[cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1]], device=device)

    zeros = torch.zeros((input_arrayy.shape[0], 1), dtype=input_arrayy.dtype, device=input_arrayy.device)
    input_arrayy = torch.cat([input_arrayy, zeros], dim=1)

    for i in range(len(random_degrees)):
        # q = R.from_euler('xyz', [0, 0, random_degrees[i]], degrees=True)
        # input_arrayl[i]=q.apply(input_arrayl[i])
        theta = random_degrees[i]  # angle of rotation in radians
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        # rotation matrix
        Rz = torch.tensor([[cos_theta, -sin_theta, 0],
                           [sin_theta, cos_theta, 0],
                           [0, 0, 1]], device=device)
        random_torch_rotation.append(Rz)
    m = input_arrayy.clone()

    for i in range(len(input_arrayy)):
        input_arrayy[i] = torch.mm(m[i].unsqueeze(0), random_torch_rotation[i])[0]
    # input_arrayy=torch.mm(input_arrayy,Rzz)
    input_arrayy = input_arrayy[:, :-1]
    # input_array=torch.tensor(input_arrayl[:,:-1],device=device)

    return input_arrayy


def featTransformationModule(feat, device):
    feat_clone = feat.clone()
    random_torch_rotation = []
    thetaa = math.pi / 4  # angle of rotation in radians
    cos_thetaa = math.cos(thetaa)
    sin_thetaa = math.sin(thetaa)
    # rotation matrix
    Rzz = torch.tensor([[cos_thetaa, -sin_thetaa, 0],
                        [sin_thetaa, cos_thetaa, 0],
                        [0, 0, 1]], device=device)

    random_degrees = [random.uniform(0, math.pi / 2) for j in range(feat.shape[0])]
    # random_degrees=[math.pi/90 for j in range (feat.shape[0])]
    # random_degrees=[]
    # degrees=[math.pi/18,math.pi/12,math.pi/6,math.pi/9]
    # for i in range (feat.shape[0]):
    #     random_degrees.append(degrees[int(i%4)])

    for i in range(feat.shape[0]):
        theta = random_degrees[i]  # angle of rotation in radians
        # print(theta)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        # rotation matrix
        Rz = torch.tensor([[cos_theta, -sin_theta, 0],
                           [sin_theta, cos_theta, 0],
                           [0, 0, 1]], device=device)
        random_torch_rotation.append(Rz)

    feat_xyz = torch.transpose(feat_clone, 1, 2)

    # pdb.set_trace()
    m = feat_xyz.clone()
    # print(feat_xyz[:,:,0:3].shape)
    # print(feat_xyz[:,:,0:3])
    for i in range(len(feat_xyz)):
        feat_xyz[i] = torch.cat(
            [torch.mm(m[i][:, 0:3], random_torch_rotation[i]), torch.mm(m[i][:, 3:], random_torch_rotation[i])], dim=1)
        # feat_xyz[i] = torch.cat([torch.mm(m[i][:, 0:3], Rzz), torch.mm(m[i][:, 3:], Rzz)], dim=1)

    # print(torch.cat([torch.transpose(feat,1,2),feat_xyz],dim=2).cpu().numpy()[0][0])
    feat_xyz_tensor = torch.transpose(feat_xyz, 1, 2)
    output_tensor = feat_xyz_tensor

    return [output_tensor, random_degrees]


def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        from data_ridi import RIDIGlobSpeedSequence
        seq_type = RIDIGlobSpeedSequence
    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion_cosine = torch.nn.CosineSimilarity(dim=1)
    criterion_cosineEmbedded = torch.nn.CosineEmbeddingLoss(margin=0.0)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    # //////////////////////////////////////////////////////////////////////////////////////////////
    # ronin_model_path = osp.join(args.out_dir, 'checkpoint_RoNIN_RIDI.pt')[1:]

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        ronin_ridi_checkpoint = torch.load(args.pretrained_model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        ronin_ridi_checkpoint = torch.load(args.pretrained_model_path)

    ronin_ridi_network = get_model(args.arch)

    ronin_ridi_network.load_state_dict(ronin_ridi_checkpoint['model_state_dict'])
    ronin_ridi_network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.pretrained_model_path, device))

    # //////////////////////////////////////////////////////////////////////////////////

    try:
        print("Pretraining Started..........")
        for epoch in range(start_epoch, args.epochs):

            # targets, preds = run_test(network, seq_loader, device, True)

            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            z = 0
            for batch_id, (feat, targ, _, _) in enumerate(train_loader):
                feat, targ = feat.to(device), targ.to(device)
                feat_copy = feat.detach().requires_grad_(False)
                pred = network(feat)
                pretrained_pred = ronin_ridi_network(feat)
                feat_contrast, random_degrees = featTransformationModule(feat_copy, device)
                feat_contrast_c = feat_contrast.detach().requires_grad_(False)

                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                feat = feat.detach().requires_grad_(False)
                pred_copy = pred.detach().requires_grad_(False)

                pred_c = targetTransformationModule(pred, random_degrees, device)

                v_2 = network(feat_contrast_c)

                for i in range(len(pred)):
                    if i == 0:
                        loss_2 = 1 - criterion_cosine(torch.unsqueeze(v_2[i], 0),
                                                      torch.unsqueeze(pred_c[i], 0)).requires_grad_(True)
                    else:
                        if torch.norm(pred_c[i]) > 0.5:
                            loss_2 += 1 - criterion_cosine(torch.unsqueeze(v_2[i], 0),
                                                           torch.unsqueeze(pred_c[i], 0)).requires_grad_(True)
                        else:
                            loss_2 += 0

                # loss_2=loss_2/len(pred)
                loss_1 = criterion(pred, pretrained_pred)
                loss_2 = criterion(v_2, pred_c)
                total_loss = loss_1 + loss_2

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                step += 1
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))
            print("navigator/total_loss: " + str(total_loss))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
            else:
                if args.out_dir is not None and osp.isdir(args.out_dir):
                    if (epoch % 20 == 0):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        model_path_neptune = "navigator/model_checkpoints/checkpoint_" + str(epoch)
                        # run[model_path_neptune].upload(model_path)
                        print('Model saved to ', model_path)

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        model_path_neptune = "navigator/model_checkpoints/checkpoint_" + str(epoch)
        # run[model_path_neptune].upload(model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test_sequence(args):
    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.root_dir
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args)

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(args.arch)

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60

    for data in test_data_list:
        seq_dataset = get_dataset(root_dir, [data], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int)

        targets, preds = run_test(network, seq_loader, device, True)
        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
        pos_gt = seq_dataset.gt_pos[0][:, :2]

        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

        # Plot figures
        kp = preds.shape[1]
        if kp == 2:
            targ_names = ['vx', 'vy']
        elif kp == 3:
            targ_names = ['vx', 'vy', 'vz']

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        for i in range(kp):
            plt.subplot2grid((kp, 2), (i, 1))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targets[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                    np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
            plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))
            model_path_neptune = "navigator/out_dir/" + str(data) + "_gsn.png"
            # run[model_path_neptune].upload(osp.join(args.out_dir, data + '_gsn.png'))
            # run[model_path_neptune].upload(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    # Export a csv file
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(test_data_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    return losses_avg


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


if __name__ == '__main__':
    # run = neptune.init_run(
    #     project="Navigator/Navigator",
    #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYTk4NGQwYS1lMWQxLTQ3YWQtYmQ3NC1lMzBjNDVmNDI3MzAifQ==",
    # )

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
    parser.add_argument('--pretrained_model_path', type=str, default=None)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    # run["parameters"] = args

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')
