from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import shutil
import tempfile
import subprocess
from typing import Literal
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pickle
from pytorch3d import transforms

# 移除了 diffusion 和 denoiser 相关的 import
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset_fd import WeightedPrimitiveSequenceDatasetV2
from utilss.smpl_utils import *
import tyro
import yaml
import cv2
from smplx import SMPLX
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mld.train_mvae import Args as MVAEArgs

# ----------------- 参数配置 -----------------

@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = None
    dataset: str = 'babel'
    
    # 修改：直接指向 VAE 的 checkpoint
    vae_checkpoint: str = '' 
    
    pkl_dir: str = './data/seq_data'
    split: str = 'test'
    batch_size: int = 1
    export_smpl: int = 1
    
    # 移除了 guidance, zero_noise 等 diffusion 参数
    
    use_predicted_joints: int = 1
    fix_floor: int = 0
    
    # 模式选择
    use_gt_motion: int = 0
    use_vae_recon: int = 1 # 默认为 1，如果想看 GT 请在命令行设为 0 并开启 use_gt_motion

# ----------------- 模型加载 -----------------

def load_vae(vae_checkpoint_path, device):
    """
    仅加载 VAE 模型
    """
    vae_checkpoint_path = Path(vae_checkpoint_path)
    vae_dir = vae_checkpoint_path.parent
    
    # 加载配置
    config_path = vae_dir / "args.yaml"
    if not config_path.exists():
        # 如果直接在 checkpoint 目录下找不到，尝试找上一级 (兼容不同的保存结构)
        config_path = vae_dir.parent / "args.yaml"
        
    print(f"Loading VAE config from {config_path}")
    with open(config_path, "r") as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
        
    # 初始化模型
    vae_model = AutoMldVae(
        **asdict(vae_args.model_args),
    ).to(device)
    
    # 加载权重
    print(f"Loading VAE checkpoint from {vae_checkpoint_path}")
    checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    
    # 处理 latent 统计量
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
        
    vae_model.load_state_dict(model_state_dict)
    vae_model.latent_mean = model_state_dict['latent_mean']
    vae_model.latent_std = model_state_dict['latent_std']
    
    print(f"latent_mean: {vae_model.latent_mean}")
    print(f"latent_std: {vae_model.latent_std}")
    
    # 冻结参数
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()
    
    return vae_args, vae_model

# ----------------- 核心逻辑 -----------------

# def rollout(seq_name, vae_args, vae_model, dataset, rollout_args):
#     device = rollout_args.device
#     batch_size = rollout_args.batch_size
#     future_length = dataset.future_length
#     history_length = dataset.history_length
#     primitive_length = history_length + future_length
    
#     # 检查数据是否存在
#     seq_data = next((data for data in dataset.dataset if data['seq_name'] == seq_name), None)
#     if seq_data is None:
#         raise ValueError(f"No sequence found with seq_name: {seq_name}")
    
#     # 计算总帧数和循环次数
#     # 注意：这里我们主要依靠 dataset 的 batch 逻辑来获取分段数据
#     # 为了简化计算，我们先获取整个序列的长度参考
#     all_music = dataset.get_full_music_sequence(seq_name)
#     num_frames = all_music.shape[0]
#     num_rollout = num_frames // future_length

#     primitive_utility = dataset.primitive_utility
#     print('body_type:', primitive_utility.body_type)
    
#     # 设置保存路径
#     out_path = rollout_args.save_dir
#     filename = f'seed{rollout_args.seed}_{seq_name}'
    
#     if rollout_args.fix_floor:
#         filename = f'fixfloor_{filename}'
#     if rollout_args.use_gt_motion:
#         filename = f'GT_{filename}'
#     elif rollout_args.use_vae_recon:
#         filename = f'VAE_Recon_{filename}'
        
#     out_path = out_path / filename
#     out_path.mkdir(parents=True, exist_ok=True)
    
#     # 获取初始 batch
#     batch = dataset.get_batch_infer()
#     input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    
#     gender = model_kwargs['y']['gender'][0]
#     betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)
    
#     pelvis_delta = primitive_utility.calc_calibrate_offset({
#         'betas': betas[:, 0, :],
#         'gender': gender,
#     })
    
#     input_motions = input_motions.to(device)
#     motion_tensor = input_motions.squeeze(2).permute(0, 2, 1) # [B, T, D]
#     history_motion_gt = motion_tensor[:, :history_length, :]
    
#     motion_sequences = None
#     history_motion = history_motion_gt
    
#     # 坐标变换矩阵初始化
#     transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
#     transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    
#     if rollout_args.fix_floor:
#         motion_dict = primitive_utility.tensor_to_dict(dataset.denormalize(history_motion_gt))
#         joints = motion_dict['joints'].reshape(batch_size, history_length, 22, 3)
#         init_floor_height = joints[:, 0, :, 2].amin(dim=-1)
#         transf_transl[:, :, 2] = -init_floor_height.unsqueeze(-1)
        
#     # --- 主循环：逐段处理 ---
#     for segment_id in tqdm(range(num_rollout-1)):
#         # 获取当前段的 GT 数据 (作为 VAE 输入或直接作为 GT 输出)
#         input_motions = batch[segment_id]['motion_tensor_normalized'].to(device)
#         motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
#         future_frames_gt = motion_tensor[:, history_length:primitive_length, :]
        
#         if rollout_args.use_gt_motion:
#             # GT 模式：直接使用真实数据
#             future_frames = dataset.denormalize(future_frames_gt)
            
#         elif rollout_args.use_vae_recon:
#             # VAE 重建模式
#             # Encode: 输入 Future GT 和 History
#             # 注意：这里 history_motion 使用上一步更新后的 (autoregressive) 还是 GT history?
#             # 通常 VAE 重建任务中，输入给 Encoder 的是 GT。
#             # 这里我们使用当前 batch 中的 GT history 来辅助 Encode，保证 Latent 准确
#             history_motion_for_encode = motion_tensor[:, :history_length, :]
            
#             latent_gt, _ = vae_model.encode(
#                 future_motion=future_frames_gt,
#                 history_motion=history_motion_for_encode,
#                 scale_latent=False # VAE 单独推理通常不需要 scale，除非训练时有特殊设定
#             )
            
#             # Decode: 使用 Latent 和 当前的 History (可能是这一路累积下来的)
#             future_motion_pred = vae_model.decode(
#                 latent_gt, 
#                 history_motion, # 这里传入的是循环中维护的 history_motion
#                 nfuture=future_length,
#                 scale_latent=False
#             )
            
#             future_frames = dataset.denormalize(future_motion_pred)
            
#         else:
#             raise ValueError("Must select either use_gt_motion or use_vae_recon")

#         # 拼接历史和当前生成的未来帧
#         all_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
        
#         if segment_id == 0:
#             future_frames = all_frames
            
#         # --- Fix Floor Logic (保持不变) ---
#         if rollout_args.fix_floor:
#             future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
#             joints = future_feature_dict['joints'].reshape(batch_size, -1, 22, 3)
#             joints = torch.einsum('bij,btkj->btki', transf_rotmat, joints) + transf_transl.unsqueeze(1)
#             min_height = joints[:, :, :, 2].amin(dim=-1)
#             transl_floor = torch.zeros(batch_size, joints.shape[1], 3, device=device, dtype=torch.float32)
#             transl_floor[:, :, 2] = - min_height
#             future_feature_dict['transl'] += transl_floor
#             transl_delta_local = torch.einsum('bij,bti->btj', transf_rotmat, transl_floor)
#             joints += transl_delta_local.unsqueeze(2)
#             future_feature_dict['joints'] = joints.reshape(batch_size, -1, 66)
#             future_frames = primitive_utility.dict_to_tensor(future_feature_dict)
            
#         # --- 转换并累积结果 ---
#         future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
#         future_feature_dict.update(
#             {
#                 'transf_rotmat': transf_rotmat,
#                 'transf_transl': transf_transl,
#                 'gender': gender,
#                 'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
#                 'pelvis_delta': pelvis_delta,
#             }
#         )
        
#         future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
#         future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
        
#         if motion_sequences is None:
#             motion_sequences = future_primitive_dict
#         else:
#             for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
#                 motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)

#         # --- 更新 History ---
#         new_history_frames = all_frames[:, -history_length:, :]
#         history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
#         history_feature_dict.update(
#             {
#                 'transf_rotmat': transf_rotmat,
#                 'transf_transl': transf_transl,
#                 'gender': gender,
#                 'betas': betas[:, :history_length, :],
#                 'pelvis_delta': pelvis_delta,
#             }
#         )
#         canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
#             history_feature_dict, use_predicted_joints=rollout_args.use_predicted_joints)
        
#         transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
#                                        canonicalized_history_primitive_dict['transf_transl']
#         history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
#         history_motion = dataset.normalize(history_motion)
        
#     # --- 保存与渲染 ---
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
        
#     for idx in range(rollout_args.batch_size):
#         sequence = {
#             'gender': motion_sequences['gender'],
#             'betas': motion_sequences['betas'][idx],
#             'transl': motion_sequences['transl'][idx],
#             'global_orient': motion_sequences['global_orient'][idx],
#             'body_pose': motion_sequences['body_pose'][idx],
#             'joints': motion_sequences['joints'][idx],
#             'history_length': history_length,
#             'future_length': future_length,
#         }
#         tensor_dict_to_device(sequence, 'cpu')
        
#         # 保存 pkl
#         with open(out_path / f'sample_{seq_name}.pkl', 'wb') as f:
#             pickle.dump(sequence, f)
            
#         # 导出并渲染
#         if rollout_args.export_smpl:
#             poses = transforms.matrix_to_axis_angle(
#                 torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
#             ).reshape(-1, 22 * 3)
#             poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
#                               dim=1)
#             data_dict = {
#                 'mocap_framerate': min(dataset.target_fps, 30),
#                 'gender': sequence['gender'],
#                 'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
#                 'poses': poses.detach().cpu().numpy(),
#                 'trans': sequence['transl'].detach().cpu().numpy(),
#             }
#             npz_path = out_path / f'sample_{seq_name}_smplx.npz'
#             with open(npz_path, 'wb') as f:
#                 np.savez(f, **data_dict)
                
#             # Render video
#             render_video_from_npz(npz_path, sequence['joints'], rollout_args)
            
#     abs_path = out_path.absolute()
#     print(f'[Done] Results are at [{abs_path}]')


def rollout(seq_name, vae_args, vae_model, dataset, rollout_args):
    device = rollout_args.device
    batch_size = rollout_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    
    # 从 dataset.dataset 中提取对应 seq_name 的 music 和初始 motion
    seq_data = next((data for data in dataset.dataset if data['seq_name'] == seq_name), None)
    if seq_data is None:
        raise ValueError(f"No sequence found with seq_name: {seq_name}")
    all_music = dataset.get_full_music_sequence(seq_name)
    num_frames = all_music.shape[0]
    num_rollout = num_frames // future_length
    all_music = all_music[:num_rollout * future_length].reshape(num_rollout, future_length, -1)  # [num_rollout, F, 35]
    
    primitive_utility = dataset.primitive_utility
    print('body_type:', primitive_utility.body_type)
    out_path = rollout_args.save_dir
    filename = f'seed{rollout_args.seed}_{seq_name}'
    if rollout_args.fix_floor:
        filename = f'fixfloor_{filename}'
    if rollout_args.use_gt_motion:
        filename = f'GT_{filename}'
    elif rollout_args.use_vae_recon:
        filename = f'VAE_Recon_{filename}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)
    batch = dataset.get_batch_infer()  # 使用修改后的 get_batch，获取首帧初始
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    # del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)  # [B, H+F, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    input_motions = input_motions.to(device)  # [B, D, 1, T]
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
    history_motion_gt = motion_tensor[:, :history_length, :]  # [B, H, D]
    motion_sequences = None
    history_motion = history_motion_gt
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    if rollout_args.fix_floor:
        motion_dict = primitive_utility.tensor_to_dict(dataset.denormalize(history_motion_gt))
        joints = motion_dict['joints'].reshape(batch_size, history_length, 22, 3)  # [B, T, 22, 3]
        init_floor_height = joints[:, 0, :, 2].amin(dim=-1)  # [B]
        transf_transl[:, :, 2] = -init_floor_height.unsqueeze(-1)
    for segment_id in tqdm(range(num_rollout-1)):
        if rollout_args.use_gt_motion:
            input_motions = batch[segment_id]['motion_tensor_normalized'].to(device)  # [B, D, 1, T]
            motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
            future_frames = motion_tensor[:, history_length:primitive_length, :]
            future_frames = dataset.denormalize(future_frames)
        elif rollout_args.use_vae_recon:
            input_motions = batch[segment_id]['motion_tensor_normalized'].to(device)  # [B, D, 1, T]
            motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
            future_frames = motion_tensor[:, history_length:primitive_length, :]
            
            latent_gt, _ = vae_model.encode(future_motion=future_frames,
                                        history_motion=history_motion,
                                        scale_latent=1)  # [T=1, B, D]
                    
            # latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_gt, history_motion, nfuture=future_length,
                                                scale_latent=1)  # [B, F, D], normalized
            future_frames = dataset.denormalize(future_motion_pred)
    
        all_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
        if segment_id == 0:  # add init history motion
            future_frames = all_frames
        if rollout_args.fix_floor:
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            joints = future_feature_dict['joints'].reshape(batch_size, -1, 22, 3)  # [B, T, 22, 3]
            joints = torch.einsum('bij,btkj->btki', transf_rotmat, joints) + transf_transl.unsqueeze(1)
            min_height = joints[:, :, :, 2].amin(dim=-1)  # [B, T]
            transl_floor = torch.zeros(batch_size, joints.shape[1], 3, device=device, dtype=torch.float32)  # [B, T, 3]
            transl_floor[:, :, 2] = - min_height
            future_feature_dict['transl'] += transl_floor
            transl_delta_local = torch.einsum('bij,bti->btj', transf_rotmat, transl_floor)
            joints += transl_delta_local.unsqueeze(2)
            future_feature_dict['joints'] = joints.reshape(batch_size, -1, 66)
            future_frames = primitive_utility.dict_to_tensor(future_feature_dict)
        future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
        future_feature_dict.update(
            {
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
                'pelvis_delta': pelvis_delta,
            }
        )
        future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
        future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
        if motion_sequences is None:
            motion_sequences = future_primitive_dict
        else:
            for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)  # [B, T, ...]
        new_history_frames = all_frames[:, -history_length:, :]
        history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
        history_feature_dict.update(
            {
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': betas[:, :history_length, :],
                'pelvis_delta': pelvis_delta,
            }
        )
        canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
            history_feature_dict, use_predicted_joints=rollout_args.use_predicted_joints)
        transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
                                       canonicalized_history_primitive_dict['transf_transl']
        history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
        history_motion = dataset.normalize(history_motion)  # [B, T, D]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for idx in range(rollout_args.batch_size):
        sequence = {
            'gender': motion_sequences['gender'],
            'betas': motion_sequences['betas'][idx],
            'transl': motion_sequences['transl'][idx],
            'global_orient': motion_sequences['global_orient'][idx],
            'body_pose': motion_sequences['body_pose'][idx],
            'joints': motion_sequences['joints'][idx],
            'history_length': history_length,
            'future_length': future_length,
        }
        tensor_dict_to_device(sequence, 'cpu')
        with open(out_path / f'sample_{seq_name}.pkl', 'wb') as f:
            pickle.dump(sequence, f)
        if rollout_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': min(dataset.target_fps, 30),
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            npz_path = out_path / f'sample_{seq_name}_smplx.npz'
            with open(npz_path, 'wb') as f:
                np.savez(f, **data_dict)
            # Render video from npz
            render_video_from_npz(npz_path, sequence['joints'], rollout_args)
    abs_path = out_path.absolute()
    print(f'[Done] Results are at [{abs_path}]')


# ----------------- 渲染相关函数 (保持不变) -----------------

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],         # Right Leg
    [0, 1, 4, 7, 10],         # Left Leg
    [0, 3, 6, 9, 12, 15],     # Spine -> Head
    [9, 14, 17, 19, 21],      # Right Arm
    [9, 13, 16, 18, 20]       # Left Arm
]

def render_video_from_npz(npz_path, all_joints, rollout_args):
    """
    Args:
        npz_path: 路径，用于生成文件名
        all_joints: 形状为 [T, 22, 3] 的 Tensor 或 Numpy
        rollout_args: 参数
    """
    print(f"Loading data from {npz_path}...")
    # 注意：这里保留了你原来的路径，请确保该路径下文件存在
    smplx_model = SMPLX(
        model_path='./data/smplx_lockedhead_20230207/models_lockedhead/smplx', 
        gender='male', 
        use_pca=False
    )
    
    data = np.load(npz_path)
    # betas = torch.from_numpy(data['betas']).float()
    # poses = torch.from_numpy(data['poses']).float()
    # trans = torch.from_numpy(data['trans']).float()
    framerate = int(data['mocap_framerate'])
    
    if isinstance(all_joints, torch.Tensor):
        all_joints = all_joints.detach().cpu().numpy()
    
    num_frames = len(all_joints)
    print(f"Total frames: {num_frames}, Joints shape: {all_joints.shape}")
    
    temp_dir = tempfile.mkdtemp()
    
    # 修改输出路径结构，放在 vae_recon 文件夹下
    video_path = Path(npz_path).parent / f'{Path(npz_path).stem}.mp4'
    
    chain_colors = ['red', 'blue', 'black', 'orange', 'purple'] 
    radius = 1.5 
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def plot_xyPlane(minx, maxx, miny, maxy, minz):
        verts = [[minx, miny, minz], [minx, maxy, minz], [maxx, maxy, minz], [maxx, miny, minz]]
        xy_plane = Poly3DCollection([verts])
        xy_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xy_plane)

    try:
        print("开始生成帧图像 (Skeleton Mode)...")
        # 限制最大帧数，防止demo时间过长，如果需要全长请去掉切片
        render_frames = min(num_frames, 300) 
        
        for i in range(render_frames):
            current_joints = all_joints[i] # [22, 3]
            
            ax.cla()
            plot_xyPlane(-radius*2, radius*2, -radius*2, radius*2, 0)
            
            for chain, color in zip(t2m_kinematic_chain, chain_colors):
                x_data = current_joints[chain, 0]
                y_data = current_joints[chain, 1] 
                z_data = current_joints[chain, 2] 
                ax.plot3D(x_data, y_data, z_data, linewidth=2.0, color=color)
            
            ax.set_xlim([-radius, radius])
            ax.set_ylim([-radius, radius])
            ax.set_zlim([-radius, radius]) 
            ax.set_axis_off()
            ax.view_init(elev=20, azim=-60)
            
            frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
            plt.savefig(frame_path, dpi=80) 
            
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{render_frames} 帧")

        print("正在调用 FFmpeg 合成视频...")
        cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'error',
            '-r', str(framerate),
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '20',
            str(video_path)
        ]
        subprocess.run(cmd, check=True)
        print(f'视频成功保存至: {video_path}')

    except Exception as e:
        print(f"生成视频时发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# ----------------- Main -----------------

if __name__ == '__main__':
    rollout_args = tyro.cli(RolloutArgs)
    
    # 随机种子设置
    random.seed(rollout_args.seed)
    np.random.seed(rollout_args.seed)
    torch.manual_seed(rollout_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = rollout_args.torch_deterministic
    
    device = torch.device(rollout_args.device if torch.cuda.is_available() else "cpu")
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)
    rollout_args.device = device
    
    # 加载 VAE
    # 注意：如果只运行 GT 模式，其实可以不加载 VAE，但为了代码统一这里还是加载了
    vae_args, vae_model = load_vae(rollout_args.vae_checkpoint, device)
    
    # 设置保存路径
    vae_checkpoint = Path(rollout_args.vae_checkpoint)
    save_dir = vae_checkpoint.parent / 'vis_eval'
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_args.save_dir = save_dir
    
    # 加载数据集
    print("Loading Dataset...")
    dataset = WeightedPrimitiveSequenceDatasetV2(cfg_path=vae_args.data_args.cfg_path,
                                                 dataset_path=vae_args.data_args.data_dir,
                                                 body_type=vae_args.data_args.body_type,
                                                 split=rollout_args.split,
                                                 device=device,
                                                 enforce_gender='male',
                                                 enforce_zero_beta=1,
                                                 inference_mode=True)
    
    seq_names = [data['seq_name'] for data in dataset.dataset]
    print(f"Found {len(seq_names)} sequences.")
    
    for seq_name in seq_names:
        # 创建临时 Dataset 获取指定序列
        temp_dataset = WeightedPrimitiveSequenceDatasetV2(cfg_path=vae_args.data_args.cfg_path,
                                                          dataset_path=vae_args.data_args.data_dir,
                                                          body_type=vae_args.data_args.body_type,
                                                          split=rollout_args.split,
                                                          device=device,
                                                          enforce_gender='male',
                                                          enforce_zero_beta=1,
                                                          inference_mode=True,
                                                          seq_name=seq_name)
        print(f'Processing sequence: {seq_name}')
        rollout(seq_name, vae_args, vae_model, temp_dataset, rollout_args)