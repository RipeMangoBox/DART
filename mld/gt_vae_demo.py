from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict, make_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from tornado.gen import sleep
from tqdm import tqdm
import pickle
import json
import copy
import pyrender
import trimesh
import threading

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset, SinglePrimitiveDataset
from utilss.smpl_utils import *
from utilss.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from visualize.vis_seq import makeLookAt
from pyrender.trackball import Trackball

debug = 0

camera_position = np.array([0.0, 5., 2.0])
up = np.array([0, 0.0, 1.0])

gender = 'male'
frame_idx = 0
text_prompt = 'stand'
text_embedding = None
motion_tensor = None

@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    batch_size: int = 1
    save_dir = None
    dataset: str = 'babel'
    device: str = 'cuda'

    denoiser_checkpoint: str = ''
    respacing: str = ''

    text_prompt: str = ''
    guidance_param: float = 1.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0

    # 模式选择：'generation' | 'gt' | 'vae'
    viz_mode: str = 'vae'


class ClassifierFreeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        # print('scale:', y['scale'])
        return out_uncond + (y['scale'] * (out - out_uncond))

def load_mld(denoiser_checkpoint, device):
    # load denoiser
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
    # load mvae model and freeze
    print('denoiser model type:', denoiser_args.model_type)
    print('denoiser model args:', asdict(denoiser_args.model_args))
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

    # load vae
    vae_checkpoint = denoiser_args.mvae_path
    vae_dir = Path(vae_checkpoint).parent
    with open(vae_dir / "args.yaml", "r") as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
    # load mvae model and freeze
    print('vae model args:', asdict(vae_args.model_args))
    vae_model = AutoMldVae(
        **asdict(vae_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_args.mvae_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(model_state_dict)
    vae_model.to(device)
    vae_model.latent_mean = model_state_dict[
        'latent_mean']  # register buffer seems to be not loaded by load_state_dict
    vae_model.latent_std = model_state_dict['latent_std']
    print(f"Loading vae checkpoint from {denoiser_args.mvae_path}")
    print(f"latent_mean: {vae_model.latent_mean}")
    print(f"latent_std: {vae_model.latent_std}")
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()

    return denoiser_args, denoiser_model, vae_args, vae_model

# ==========================================
# 2. 辅助函数：加载下一个样本 (核心逻辑修改)
# ==========================================
def load_next_sample(viz_mode):
    """
    根据模式从 Dataset 获取数据，并更新全局变量 motion_tensor
    """
    global motion_tensor, gender, betas, pelvis_delta, text_prompt

    # 1. 从数据集获取一个 Batch
    # 注意：get_batch 内部通常是随机采样的
    batch = dataset.get_batch(batch_size=rollout_args.batch_size)
    
    # 提取基础信息
    model_kwargs = {'y': batch[0]}
    gender = model_kwargs['y']['gender'][0]
    
    # 提取 Motion (Normalized) [B, D, 1, T]
    input_motions_norm = batch[0]['motion_tensor_normalized'].to(device)
    
    # 获取对应文本（如果有）打印出来方便观察
    if 'text' in batch[0]:
        print(f"Sample Text: {batch[0]['text'][0]}")

    # 准备 SMPL 参数 (取整个序列长度)
    T_total = input_motions_norm.shape[-1]
    betas = model_kwargs['y']['betas'][:, :T_total, :].to(device)
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })

    # 2. 根据模式处理数据
    if viz_mode == 'gt':
        # GT 模式：直接反归一化
        motion_norm = input_motions_norm.squeeze(2).permute(0, 2, 1) # [B, T, D]
        motion_tensor = dataset.denormalize(motion_norm)
        print(f"Loaded GT sample (Length: {T_total})")

    elif viz_mode == 'vae':
        # VAE 模式：Encode -> Decode -> Denormalize
        print("Running VAE reconstruction...")
        with torch.no_grad():
            # input shape to encoder: [B, D, 1, T]
            dist = vae_model.encode(input_motions_norm)
            z = dist.sample()
            # decode output shape: [B, D, 1, T]
            recon_norm = vae_model.decode(z, length=T_total)
        
        recon_norm = recon_norm.squeeze(2).permute(0, 2, 1) # [B, T, D]
        motion_tensor = dataset.denormalize(recon_norm)
        print(f"Reconstructed VAE sample (Length: {T_total})")
    
    # 重置播放进度
    global frame_idx
    frame_idx = 0
    
def rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args):
    global motion_tensor
    sample_fn = diffusion.p_sample_loop if rollout_args.respacing == '' else diffusion.ddim_sample_loop
    guidance_param = torch.ones(RolloutArgs.batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * rollout_args.guidance_param
    history_motion_tensor = motion_tensor[:, -history_length:, :]  # [B, H, D]
    # canonicalize history motion
    history_feature_dict = primitive_utility.tensor_to_dict(history_motion_tensor)
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(RolloutArgs.batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(RolloutArgs.batch_size, 1, 1)
    history_feature_dict.update({
        'transf_rotmat': transf_rotmat,
        'transf_transl': transf_transl,
        'gender': gender,
        'betas': betas[:, :history_length, :],
        'pelvis_delta': pelvis_delta,
    })
    canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
        history_feature_dict, use_predicted_joints=rollout_args.use_predicted_joints)
    transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
        canonicalized_history_primitive_dict['transf_transl']
    history_motion_normalized = dataset.normalize(primitive_utility.dict_to_tensor(blended_feature_dict))

    y = {
        'text_embedding': text_embedding,
        'history_motion_normalized': history_motion_normalized,
        'scale': guidance_param,
    }

    x_start_pred = sample_fn(
        denoiser_model,
        (RolloutArgs.batch_size, *denoiser_args.model_args.noise_shape),
        clip_denoised=False,
        model_kwargs={'y': y},
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=torch.zeros_like(guidance_param) if rollout_args.zero_noise else None,
        const_noise=False,
    )  # [B, T=1, D]
    latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
    future_motion_pred = vae_model.decode(latent_pred, history_motion_normalized, nfuture=future_length,
                                               scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

    future_frames = dataset.denormalize(future_motion_pred)
    future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
    future_feature_dict.update(
        {
            'transf_rotmat': transf_rotmat,
            'transf_transl': transf_transl,
            'gender': gender,
            'betas': betas[:, :future_length, :],
            'pelvis_delta': pelvis_delta,
        }
    )
    future_feature_dict = primitive_utility.transform_feature_to_world(future_feature_dict)
    future_tensor = primitive_utility.dict_to_tensor(future_feature_dict)
    motion_tensor = torch.cat([motion_tensor, future_tensor], dim=1)  # [B, T+F, D]



def read_input():
    global text_prompt
    global text_embedding
    global motion_tensor
    while True:
        user_input = input()
        print(f"You entered new prompt: {user_input}")
        text_prompt = user_input
        text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(dtype=torch.float32,
                                                                                              device=device)  # [1, 512]
        motion_tensor = motion_tensor[:, :frame_idx + 1, :]
        if user_input.lower() == "exit":
            print("Exit")
            break

def get_body():
    motion_feature_dict = primitive_utility.tensor_to_dict(motion_tensor[:, frame_idx:frame_idx+1, :])
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(RolloutArgs.batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(RolloutArgs.batch_size, 1, 1)
    motion_feature_dict.update(
        {
            'transf_rotmat': transf_rotmat,
            'transf_transl': transf_transl,
            'gender': gender,
            'betas': betas[:, :1, :],
            'pelvis_delta': pelvis_delta,
        }
    )
    smpl_dict = primitive_utility.feature_dict_to_smpl_dict(motion_feature_dict)
    for key in ['transl', 'global_orient', 'body_pose', 'betas']:
        smpl_dict[key] = smpl_dict[key][0]
    output = body_model(return_verts=True, **smpl_dict)
    vertices = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0].detach().cpu().numpy()
    return vertices, joints, body_model.faces

def generate():
    global frame_idx
    while True:
        if frame_idx >= motion_tensor.shape[1]:
            rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
        if text_prompt.lower() == "exit":
            break


def start(viz_mode):
    # --- PyRender 初始化 ---
    scene = pyrender.Scene()
    camera = pyrender.camera.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = makeLookAt(position=camera_position, target=np.array([0.0, 0, 0]), up=up)
    camera_node = pyrender.Node(camera=camera, name='camera', matrix=camera_pose)
    scene.add_node(camera_node)
    
    axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
    scene.add_node(axis_node)
    
    vertices, joints, faces = get_body() # 初始 Body
    floor_height = vertices[:, 2].min()
    floor = trimesh.creation.box(extents=np.array([50, 50, 0.01]),
                                 transform=np.array([[1.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0], [0.0, 0.0, 1.0, floor_height - 0.005], [0.0, 0.0, 0.0, 1.0]]))
    floor.visual.vertex_colors = [0.8, 0.8, 0.8]
    scene.add_node(pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor'))
    
    body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=vertices, faces=faces), smooth=False), name='body')
    scene.add_node(body_node)
    
    camera_pose = makeLookAt(position=camera_position, target=joints[0], up=up)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True, viewport_size=(1920, 1920), record=False)
    viewer.render_lock.acquire()
    viewer._camera_node.matrix = camera_pose # 直接设置矩阵
    viewer._trackball = Trackball(camera_pose, viewer.viewport_size, 1.0)
    viewer._trackball._scale = 1500.0
    viewer.render_lock.release()
    
    # --- 交互提示 ---
    print('*' * 40)
    print(f"Mode: {viz_mode.upper()}")
    if viz_mode == 'generation':
        print("Input text prompt to generate new motion.")
    else:
        print("Input 'next' (or any text) to load next random sample from dataset.")
    print("Input 'exit' to quit.")
    print('*' * 40)

    # 启动输入线程
    input_thread = threading.Thread(target=read_input)
    input_thread.start()

    sleep_time = 1 / 30.0
    global frame_idx
    global text_prompt
    
    # 记录上一次处理的 prompt，用于检测输入变化
    last_prompt = text_prompt 

    while True:
        # 1. 检测输入变化，触发数据更新
        if text_prompt != last_prompt:
            last_prompt = text_prompt
            if text_prompt.lower() == "exit":
                break
            
            # 根据模式决定行为
            if viz_mode == 'generation':
                # 生成模式：调用 rollout (原有逻辑)
                # rollout 内部会使用新的 text_embedding 生成并更新 motion_tensor
                rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
                frame_idx = 0 
            else:
                # GT/VAE 模式：加载下一个 Batch
                print("Loading next sample...")
                load_next_sample(viz_mode)

        # 2. 渲染 Mesh 更新
        vertices, joints, faces = get_body()
        body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # viewer.render_lock.acquire()
        scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh, smooth=False), name='body')
        scene.add_node(body_node)
        
        camera_pose = makeLookAt(position=camera_position, target=joints[0], up=up)
        camera_pose_current = viewer._camera_node.matrix
        camera_pose_current[:, :] = camera_pose
        # viewer._trackball = Trackball(camera_pose_current, viewer.viewport_size, 1.0)
        # viewer._trackball._scale = 1500.0
        # viewer.render_lock.release()

        frame_idx += 1
        
        # 3. 播放循环逻辑
        if frame_idx >= motion_tensor.shape[1]:
            # GT/VAE 模式下：循环播放当前样本
            # 生成模式下：如果想循环播放当前生成的，也重置为0；
            # (之前的逻辑是播放完自动生成下一个，这里改为循环播放，等待用户输入新文本)
            frame_idx = 0 
            # 如果你希望生成模式播放完自动重新生成，可以解开下面注释：
            # if viz_mode == 'generation': rollout(...); frame_idx=0

        time.sleep(sleep_time)

    viewer.close_external()
    input_thread.join()
        
# ==========================================
# 4. Main 函数
# ==========================================
if __name__ == '__main__':
    rollout_args = tyro.cli(RolloutArgs)
    
    # 1. 环境初始化
    random.seed(rollout_args.seed)
    np.random.seed(rollout_args.seed)
    torch.manual_seed(rollout_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = rollout_args.torch_deterministic
    device = torch.device(rollout_args.device if torch.cuda.is_available() else "cpu")
    rollout_args.device = device

    # 2. 加载 SMPL 模型
    print("Loading SMPL model...")
    body_model = smplx.build_layer(body_model_dir, model_type='smplx',
                                   gender='male', ext='npz',
                                   num_pca_comps=12).to(device).eval()

    # 3. 加载 MLD (Denoiser + VAE)
    print("Loading MLD models...")
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(rollout_args.denoiser_checkpoint, device)
    
    # 设置 Diffusion
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = rollout_args.respacing
    diffusion = create_gaussian_diffusion(diffusion_args)

    # 4. 加载完整 Dataset
    print(f"Loading Dataset ({rollout_args.dataset})...")
    # 注意：这里和训练时一样加载整个数据集
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,
                                     dataset_path=vae_args.data_args.data_dir,
                                     sequence_path=f'./data/stand.pkl' if rollout_args.dataset == 'babel' else f'./data/stand_20fps.pkl',
                                     batch_size=rollout_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1)
    
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)

    # 初始化第一个样本
    if rollout_args.viz_mode == 'generation':
        # 生成模式初始化 (使用默认 prompt)
        text_prompt = rollout_args.text_prompt if rollout_args.text_prompt else "a person is walking"
        print(f"Initializing Generation with prompt: {text_prompt}")
        
        # 为了 rollout 能运行，我们需要初始化一些 tensor 结构
        # 这里复用 get_batch 只是为了获取 shape 和 basic info
        future_length = dataset.future_length
        history_length = dataset.history_length
        batch = dataset.get_batch(batch_size=rollout_args.batch_size)
        input_motions = batch[0]['motion_tensor_normalized'].to(device)
        model_kwargs = {'y': batch[0]}
        
        gender = model_kwargs['y']['gender'][0]
        betas = model_kwargs['y']['betas'].to(device)
        pelvis_delta = primitive_utility.calc_calibrate_offset({'betas': betas[:, 0, :], 'gender': gender})
        
        # 设置 motion_tensor 为历史帧，供 rollout 使用
        motion_tensor = dataset.denormalize(input_motions.squeeze(2).permute(0, 2, 1)[:, :history_length, :])
        text_embedding = encode_text(dataset.clip_model, [text_prompt], force_empty_zero=True).to(dtype=torch.float32, device=device)
        
        # 立即生成第一次
        rollout(denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)

    else:
        # GT / VAE 模式初始化
        print(f"Initializing {rollout_args.viz_mode.upper()} mode...")
        text_prompt = "Initial Load"
        load_next_sample(rollout_args.viz_mode)

    # 6. 开始渲染循环
    start(rollout_args.viz_mode)