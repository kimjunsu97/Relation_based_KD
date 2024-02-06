import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import tqdm

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.h36m_dataset import Human36mDataset
from time import time
from torchsummary import summary
from common.utils import deterministic_random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gc
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids = [1, 2, 3]

args = parse_args()
# dataset 준비
#####################################
print('Loading dataset...')
dataset_path = 'data/data_3d_h36m.npz'
dataset = Human36mDataset(dataset_path)
print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue
            
        for cam_idx in range(len(keypoints[subject][action])):
            
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    

    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None 
    
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)
causal_shift = 0
# model teacher
#######################################################
filter_widths_teacher = [int(x) for x in args.architecture_teacher.split(',')]    
filter_widths_student = [int(x) for x in args.architecture_student.split(',')]    

# filter_widths_729 = [3,3,3,3,3,3]    
# filter_widths_243 = [3,3,3,3,3]    
# filter_widths_81 = [3,3,3,3]    
# filter_widths_27 = [3,3,3]    

# filter_width_teacher = args.filter_widths_teacher
# filter_width_student = args.filter_widths_student

model_teacher = TemporalModel(17, 2, 17,
                            filter_widths=filter_widths_teacher, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
#checkpoint_teacher = torch.load("D:/paper/paper/checkpoint/teacher/gt_243_epoch_100.bin")
checkpoint_teacher = torch.load(os.path.join(args.checkpoint_teacher, args.teacher_name))

model_teacher.load_state_dict(checkpoint_teacher['model_pos'])
receptive_field_teacher = model_teacher.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field_teacher))
pad_teacher = (receptive_field_teacher - 1) // 2 # Padding on each side

model_params_teacher = 0
for parameter in model_teacher.parameters():
    model_params_teacher += parameter.numel()
print('INFO: Trainable parameter count:', model_params_teacher)

# model student
########################################################
model_student = TemporalModel(17, 2, 17,
                            filter_widths=filter_widths_student, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
#checkpoint_student = torch.load("/home/ssu40/test/KD/checkpoint/gtstudent/epoch_100.bin")
#model_student.load_state_dict(checkpoint_student['model_pos'])
receptive_field_student = model_student.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field_student))
pad_student = (receptive_field_student - 1) // 2 # Padding on each side

model_params_student = 0
for parameter in model_student.parameters():
    model_params_student += parameter.numel()
print('INFO: Trainable parameter count:', model_params_student)

if torch.cuda.is_available():
    # if torch.cuda.device_count() > 1:
    #     device_ids = [1, 2, 3]
    #     model_teacher = torch.nn.DataParallel(model_teacher, device_ids=device_ids)
    #     model_student = torch.nn.DataParallel(model_student, device_ids=device_ids)
        
    model_teacher = model_teacher.to(device)
    model_student = model_student.to(device)
#########################################################

cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

train_generator_teacher = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad_teacher, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Training on {} frames'.format(train_generator_teacher.num_frames()))
train_generator_student = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad_student, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

print('INFO: Training on {} frames'.format(train_generator_student.num_frames()))

train_generator_eval_student = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad_student, causal_shift=causal_shift, augment=False)

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad_student, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(train_generator_teacher.num_frames()))
print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
print('** The final evaluation will be carried out after the last training epoch.')

lr = args.learning_rate
optimizer = optim.Adam(model_student.parameters(), lr=lr, amsgrad=True)
lr_decay = args.lr_decay

losses_3d_train = []
losses_3d_train_eval = []
losses_3d_valid = []
losses_3d_KD = []

epoch = 0
initial_momentum = 0.1
final_momentum = 0.001

# for _, batch_3d, batch_2d in train_generator_student.next_epoch():
#     print(batch_3d[300][0])
#     break
# for _, batch_3d, batch_2d in train_generator_teacher.next_epoch():
#     print(batch_3d[300][0])    
#     break
total_size = train_generator_student.num_batches

while epoch < args.epochs:
    epoch +=1
    start_time = time()
    epoch_loss_3d_train = 0
    epoch_loss_traj_train = 0
    epoch_loss_2d_train_unlabeled = 0
    N = 0
    N_semi = 0
    model_student.train()
    with tqdm.tqdm(total=total_size) as pbar:

        for (_, batch_3d_student, batch_2d_student), (_, batch_3d_teacher, batch_2d_teacher) in zip(train_generator_student.next_epoch(), train_generator_teacher.next_epoch()):
                inputs_3d_student = torch.from_numpy(batch_3d_student.astype('float32'))
                inputs_2d_student = torch.from_numpy(batch_2d_student.astype('float32'))

                inputs_3d_teacher = torch.from_numpy(batch_3d_teacher.astype('float32'))
                inputs_2d_teacher = torch.from_numpy(batch_2d_teacher.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d_student = inputs_3d_student.to(device)
                    inputs_2d_student = inputs_2d_student.to(device)
                    inputs_3d_teacher = inputs_3d_teacher.to(device)
                    inputs_2d_teacher = inputs_2d_teacher.to(device)
                inputs_3d_student[:, :, 0] = 0
                inputs_3d_teacher[:, :, 0] = 0

                optimizer.zero_grad()

                # Predict 3D poses
                predicted_3d_pos_student = model_student(inputs_2d_student)
                predicted_3d_pos_teacher = model_teacher(inputs_2d_teacher)

                loss_3d_pos_student = mpjpe(predicted_3d_pos_student, inputs_3d_student)
                loss_3d_pos_student_teacher = mpjpe(predicted_3d_pos_teacher, predicted_3d_pos_student)
                loss_3d_pos_student_teacher_angle = angle_KD(predicted_3d_pos_teacher, predicted_3d_pos_student)/args.lambda_angle
                loss_3d_pos_student_teacher_distance = distance_KD(predicted_3d_pos_teacher, predicted_3d_pos_student)/args.lambda_distance

                epoch_loss_3d_train += inputs_3d_student.shape[0]*inputs_3d_student.shape[1] * loss_3d_pos_student.item()
                N += inputs_3d_student.shape[0]*inputs_3d_student.shape[1]
                # print("student : ",loss_3d_pos_student)
                # print("student_teacher : ",loss_3d_pos_student_teacher)
                # print("student_teacher angle : ",loss_3d_pos_student_teacher_angle)
                # print("student_teacher_distance : ",loss_3d_pos_student_teacher_distance)            
                loss_total = loss_3d_pos_student + loss_3d_pos_student_teacher+ loss_3d_pos_student_teacher_distance + loss_3d_pos_student_teacher_angle
                loss_total.backward()

                optimizer.step()
                pbar.update(1)
    # Evaluate
    def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        time_sum = 0
        start = time()
        with torch.no_grad():
            if not use_trajectory_model:
                model_student.eval()
            else:
                model_student.eval()
            N = 0
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.to(device)

                # Positional model
                if not use_trajectory_model:
                    predicted_3d_pos = model_student(inputs_2d)
                else:
                    predicted_3d_pos = model_traj(inputs_2d)

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    if not use_trajectory_model:
                        predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                    
                if return_predictions:
                    return predicted_3d_pos.squeeze(0).cpu().numpy()
                    
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.to(device)
                inputs_3d[:, :, 0] = 0    
                if test_generator.augment_enabled():
                    inputs_3d = inputs_3d[:1]

                error = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]
                
                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                # Compute velocity error
                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
        end = time()

        e1 = (epoch_loss_3d_pos / N)*1000
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        e3 = (epoch_loss_3d_pos_scale / N)*1000
        ev = (epoch_loss_3d_vel / N)*1000
        delay = ((end-start)/N)*1000


        return e1, e2, e3, ev, delay

    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_poses_3d, out_poses_2d

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        sum_time = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad_student, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            e1, e2, e3, ev, delay = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            sum_time.append(delay)
        print(f'[{epoch}]',end=' ')
        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm',end=',')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm',end=',')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')

    run_evaluation(all_actions, action_filter)
    
    if not os.path.exists(os.path.join(args.checkpoint_student, f'gt_{receptive_field_student}_{receptive_field_teacher}_distance_{args.lambda_distance}_angle_{args.lambda_angle}')):
        os.makedirs(os.path.join(args.checkpoint_student, f'gt_{receptive_field_student}_{receptive_field_teacher}_distance_{args.lambda_distance}_angle_{args.lambda_angle}'))

    # Save checkpoint if necessary
    chk_path = os.path.join(args.checkpoint_student, f'gt_{receptive_field_student}_{receptive_field_teacher}_distance_{args.lambda_distance}/epoch_{epoch}.bin')
    print('Saving checkpoint to', chk_path)
    
    torch.save({
        'epoch': epoch,
        'lr': lr,
        'random_state': train_generator_student.random_state(),
        'optimizer': optimizer.state_dict(),
        'model_pos': model_student.state_dict(),
        'model_traj':  None,
        'random_state_semi': None,
    }, chk_path)
    