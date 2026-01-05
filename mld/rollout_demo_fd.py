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
from diffusion import gaussian_diffusion as gd
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset_fd import WeightedPrimitiveSequenceDatasetV2  # 使用修改后的类
from utilss.smpl_utils import *
import tyro
import yaml
import cv2
from smplx import SMPLX
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from visualize.vis_seq import makeLookAt
from pyrender.trackball import Trackball

debug = 0

@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = None
    dataset: str = 'babel'
    denoiser_checkpoint: str = ''
    respacing: str = ''
    pkl_dir: str = './data/seq_data'  # pkl 文件夹（包含 split.pkl 或单个 pkl）
    split: str = 'test'  # 用于加载数据集的 split
    batch_size: int = 1  # demo 中设为 1，避免重复
    guidance_param: float = 1.0
    export_smpl: int = 1
    zero_noise: int = 0
    use_predicted_joints: int = 0
    fix_floor: int = 0
    use_gt_motion: int = 1
    use_vae_recon: int = 0
    
class ClassifierFreeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'] * (out - out_uncond))

def load_mld(denoiser_checkpoint, device):
    # 同原代码
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
    denoiser_class = DenoiserMLP if isinstance(denoiser_args.model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(
        **asdict(denoiser_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_checkpoint, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    print(f"Loading denoiser checkpoint from {denoiser_checkpoint}")
    denoiser_model.load_state_dict(model_state_dict)
    denoiser_model.to(device)
    for param in denoiser_model.parameters():
        param.requires_grad = False
    denoiser_model.eval()
    denoiser_model = ClassifierFreeWrapper(denoiser_model)
    vae_checkpoint = denoiser_args.mvae_path
    vae_dir = Path(vae_checkpoint).parent
    with open(vae_dir / "args.yaml", "r") as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
    vae_model = AutoMldVae(
        **asdict(vae_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_args.mvae_path)
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(model_state_dict)
    vae_model.latent_mean = model_state_dict['latent_mean']
    vae_model.latent_std = model_state_dict['latent_std']
    print(f"Loading vae checkpoint from {denoiser_args.mvae_path}")
    print(f"latent_mean: {vae_model.latent_mean}")
    print(f"latent_std: {vae_model.latent_std}")
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()
    return denoiser_args, denoiser_model, vae_args, vae_model

def rollout(seq_name, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args):
    device = rollout_args.device
    batch_size = rollout_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    sample_fn = diffusion.p_sample_loop if rollout_args.respacing == '' else diffusion.ddim_sample_loop
    
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
    filename = f'guidance{rollout_args.guidance_param}_seed{rollout_args.seed}_{seq_name}'
    if rollout_args.respacing != '':
        filename = f'{rollout_args.respacing}_{filename}'
    if rollout_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if rollout_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    if rollout_args.fix_floor:
        filename = f'fixfloor_{filename}'
    if rollout_args.use_gt_motion:
        filename = f'use_gt_motion_{filename}'
    if rollout_args.use_vae_recon:
        filename = f'use_vae_recon_{filename}'
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
                                        history_motion=history_motion if denoiser_args.train_rollout_history == "gt" else history_motion,
                                        scale_latent=denoiser_args.rescale_latent)  # [T=1, B, D]
                    
            # latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_gt, history_motion, nfuture=future_length,
                                                scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized
            future_frames = dataset.denormalize(future_motion_pred)
        else:
            music_embedding = all_music[segment_id].expand(batch_size, -1, -1)  # [B, F, 35]
            guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * rollout_args.guidance_param
            y = {
                'music': music_embedding,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }
            x_start_pred = sample_fn(
                denoiser_model,
                (batch_size, *denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=torch.zeros_like(guidance_param) if rollout_args.zero_noise else None,
                const_noise=False,
            )  # [B, T=1, D]
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized
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

import os
import shutil
import tempfile
import subprocess
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from smplx import SMPLX

# 如果在服务器运行，请取消注释下一行
# matplotlib.use('Agg')

# ----------------- 定义运动链 (Kinematic Chains) -----------------
# 假设这些变量在你的环境中已定义，为了代码完整性，这里提供标准 t2m 定义
# 对应 SMPL 前 22 个关节的连接关系
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],         # Right Leg
    [0, 1, 4, 7, 10],         # Left Leg
    [0, 3, 6, 9, 12, 15],     # Spine -> Head
    [9, 14, 17, 19, 21],      # Right Arm
    [9, 13, 16, 18, 20]       # Left Arm
]

# 注意：由于你指定只使用前 22 个关节，左右手的详细手指链 (t2m_left/right_hand_chain) 
# 通常涉及索引 > 21 的关节。因此在这里我们只绘制身体主躯干。
# 如果你的 t2m_left_hand_chain 定义在 22 个关节以内（不太常见），请自行添加。

def render_video_from_npz(npz_path, all_joints, rollout_args):
    """
    Args:
        npz_path: 路径，用于生成文件名
        all_joints: 形状为 [T, 22, 3] 的 Tensor 或 Numpy
        rollout_args: 参数
    """
    # ---------------- 1. 数据与模型加载 ----------------
    print(f"Loading data from {npz_path}...")
    smplx_model = SMPLX(
        model_path='./data/smplx_lockedhead_20230207/models_lockedhead/smplx', 
        gender='male', 
        use_pca=False
    )
    
    data = np.load(npz_path)
    betas = torch.from_numpy(data['betas']).float()
    poses = torch.from_numpy(data['poses']).float()
    trans = torch.from_numpy(data['trans']).float()
    framerate = int(data['mocap_framerate'])
    
    # **关键修复 1**: 确保数据是 Numpy 格式且在 CPU 上
    if isinstance(all_joints, torch.Tensor):
        all_joints = all_joints.detach().cpu().numpy()
    
    num_frames = len(all_joints)
    print(f"Total frames: {num_frames}, Joints shape: {all_joints.shape}")
    
    # ---------------- 2. 准备临时环境 ----------------
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时帧目录: {temp_dir}")
    
    video_path = f'mld_denoiser/mlp_mnorm/animations/{Path(npz_path).stem}.mp4'
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # ---------------- 3. 绘图设置 ----------------
    # 设置绘图颜色等
    # 这里的颜色列表对应上面的 5 条链
    chain_colors = ['red', 'blue', 'black', 'orange', 'purple'] 
    
    # 坐标轴范围半径 (根据动作幅度调整，通常 1-2 米)
    radius = 1.5 
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def plot_xyPlane(minx, maxx, miny, maxy, minz):
        """绘制地板灰底"""
        verts = [[minx, miny, minz], [minx, maxy, minz], [maxx, maxy, minz], [maxx, miny, minz]]
        xy_plane = Poly3DCollection([verts])
        xy_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xy_plane)

    try:
        print("开始生成帧图像 (Skeleton Mode)...")
        for i in range(180):
            # # --- A. 模型推理 ---
            # output = smplx_model(
            #     betas=betas.unsqueeze(0), 
            #     body_pose=poses[i, 3:66].reshape(1, -1, 3), 
            #     global_orient=poses[i, :3].reshape(1, -1, 3), 
            #     transl=trans[i].unsqueeze(0)
            # )
            
            # # [1, 127, 3] -> [22, 3]
            # # 仅仅提取前 22 个关节用于 t2m 格式绘制
            # joints = output.joints[0, :22].detach().cpu().numpy() 
            current_joints = all_joints[i] # Shape: [22, 3]
            
            # 坐标转换：通常 matplotlib 3D 的 Z 轴是向上的，
            # SMPL 的坐标系通常 Y 轴向上或 Z 轴向前，需要根据实际视觉效果调整。
            # 这里假设 SMPL 输出是 Y-up，我们需要将 Y 轴数据映射到 Matplotlib 的 Z 轴，
            # 或者直接绘制，使用 ax.view_init 调整视角。
            # 下面代码保持原始数据，通过 view_init 控制视角。
            
            # --- B. 绘图 ---
            ax.cla()
            
            # 1. 绘制地板
            plot_xyPlane(-radius*2, radius*2, -radius*2, radius*2, 0)
            
            # 2. 绘制骨架连线
            for chain, color in zip(t2m_kinematic_chain, chain_colors):
                x_data = current_joints[chain, 0]
                # y_data = current_joints[chain, 2] # 深度
                # z_data = -current_joints[chain, 1] # 高度
                y_data = current_joints[chain, 1] # 深度
                z_data = current_joints[chain, 2] # 高度
                
                ax.plot3D(x_data, y_data, z_data, linewidth=2.0, color=color)
            
            # --- C. 视图设置 ---
            ax.set_xlim([-radius, radius])
            ax.set_ylim([-radius, radius])
            ax.set_zlim([-radius, radius]) # 保持比例一致
            
            # 隐藏坐标轴刻度，更美观
            ax.set_axis_off()
            
            # 设置视角：elev (仰角), azim (方位角)
            # 根据需要调整，这里设为常见的斜视
            ax.view_init(elev=20, azim=-60)
            
            # 保存当前帧
            frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
            plt.savefig(frame_path, dpi=80) # dpi=80 既快又清晰
            
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{num_frames} 帧")

        # ---------------- 4. FFmpeg 合成 ----------------
        print("正在调用 FFmpeg 合成视频...")
        cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'error',
            '-r', str(framerate),
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '20',
            video_path
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
            print("临时目录已清理")

if __name__ == '__main__':
    rollout_args = tyro.cli(RolloutArgs)
    random.seed(rollout_args.seed)
    np.random.seed(rollout_args.seed)
    torch.manual_seed(rollout_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = rollout_args.torch_deterministic
    
    device = torch.device(rollout_args.device if torch.cuda.is_available() else "cpu")
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)
    
    rollout_args.device = device
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(rollout_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(rollout_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'rollout'
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_args.save_dir = save_dir
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = rollout_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)
    # 加载数据集 (inference_mode=True, 无需指定 sequence_path)
    dataset = WeightedPrimitiveSequenceDatasetV2(cfg_path=vae_args.data_args.cfg_path,
                                                 dataset_path=vae_args.data_args.data_dir,
                                                 body_type=vae_args.data_args.body_type,
                                                 split=rollout_args.split,
                                                 device=device,
                                                 enforce_gender='male',
                                                 enforce_zero_beta=1,
                                                 inference_mode=True)  # 不指定 seq_name，这里加载所有
    # 获取所有 seq_name (从 dataset.dataset)
    seq_names = [data['seq_name'] for data in dataset.dataset]
    for seq_name in seq_names:
        # 为每个 seq_name 创建临时 dataset 以过滤
        temp_dataset = WeightedPrimitiveSequenceDatasetV2(cfg_path=vae_args.data_args.cfg_path,
                                                          dataset_path=vae_args.data_args.data_dir,
                                                          body_type=vae_args.data_args.body_type,
                                                          split=rollout_args.split,
                                                          device=device,
                                                          enforce_gender='male',
                                                          enforce_zero_beta=1,
                                                          inference_mode=True,
                                                          seq_name=seq_name)
        print(f'Generating for sequence: {seq_name}')
        rollout(seq_name, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, temp_dataset, rollout_args)