import torch

def lerp(val1, val2, weight):
    """标准线性插值"""
    return (1.0 - weight) * val1 + weight * val2

def slerp_vector(v1, v2, weight):
    """对归一化的 2D 向量进行球面线性插值 (Eq 9 中的 I 操作)"""
    dot = torch.sum(v1 * v2, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # 角度极小时退化为线性插值，防止除零
    mask = (sin_theta < 1e-6).float()
    res_lerp = (1.0 - weight) * v1 + weight * v2
    res_slerp = (torch.sin((1.0 - weight) * theta) / (sin_theta + 1e-8)) * v1 + \
                (torch.sin(weight * theta) / (sin_theta + 1e-8)) * v2
    return (mask * res_lerp + (1 - mask) * res_slerp)

def rotate_2d_vector(v, theta):
    """执行 2D 旋转矩阵操作 (Eq 9 中的 R(theta) 操作)"""
    # v: [..., 2], theta: [..., 1]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x, y = v[..., 0:1], v[..., 1:2]
    # 顺时针旋转逻辑 (DeepPhase 标准)
    nx = x * cos_t - y * sin_t
    ny = x * sin_t + y * cos_t
    return torch.cat([nx, ny], dim=-1)

def interpolate(manifold_hist, manifold_pred, latent_dim, manifold_type='phase', delta_t=1/30.0, weight=0.5):
    """
    manifold_hist: 上一轮产生的流形 [B, dim, 1]
    manifold_pred: Diffusion 预测的当前轮流形 [B, dim, 1]
    """
    # 鲁棒性处理：首帧运行没有历史记录，直接返回预测值
    if manifold_hist is None:
        return manifold_pred
    
    # 分解流形分量
    parts_h = manifold_hist.split(latent_dim, dim=1)
    parts_p = manifold_pred.split(latent_dim, dim=1)

    if manifold_type == 'phase':
        # 模式 1: phase (sx, sy, f, a, b) [cite: 227]
        sx_h, sy_h, f_h, a_h, b_h = parts_h
        sx_p, sy_p, f_p, a_p, b_p = parts_p

        # 1. 构造历史相位向量并归一化
        v_h = torch.cat([sx_h, sy_h], dim=-1)
        v_h = v_h / (torch.norm(v_h, dim=-1, keepdim=True) + 1e-8)

        # 2. 旋转外推 (Rotation Prior): 根据历史频率计算逻辑上的“下一帧”位置
        # theta = 2 * PI * Frequency * delta_t [cite: 275, 277]
        theta = 2 * torch.pi * f_h * delta_t
        v_prior = rotate_2d_vector(v_h, theta)

        # 3. 预测向量归一化
        v_pred = torch.cat([sx_p, sy_p], dim=-1)
        v_pred = v_pred / (torch.norm(v_pred, dim=-1, keepdim=True) + 1e-8)

        # 4. 执行球面插值 (Slerp)
        v_final = slerp_vector(v_prior, v_pred, weight)
        v_final = v_final / (torch.norm(v_final, dim=-1, keepdim=True) + 1e-8)
        sx_final, sy_final = v_final.split(1, dim=-1)

        # 5. 形态参数插值 (f, a, b)
        f_final = lerp(f_h, f_p, weight)
        a_final = lerp(a_h, a_p, weight)
        b_final = lerp(b_h, b_p, weight)

        return torch.cat([sx_final, sy_final, f_final, a_final, b_final], dim=1)

    elif manifold_type == 'P':
        # 模式 2: P (p, f, a, b) 其中 p 为 1D 标量相位 [0, 1]
        p_h, f_h, a_h, b_h = parts_h
        p_p, f_p, a_p, b_p = parts_p

        # 1. 标量相位外推: p_new = (p_old + f * dt) mod 1 
        p_prior_val = (p_h + f_h * delta_t) % 1.0

        # 2. 为了平滑处理环形相位，将其映射到 2D 向量空间进行插值 (Slerp)
        v_prior = torch.cat([torch.sin(2 * torch.pi * p_prior_val), 
                             torch.cos(2 * torch.pi * p_prior_val)], dim=-1)
        v_pred = torch.cat([torch.sin(2 * torch.pi * p_p), 
                            torch.cos(2 * torch.pi * p_p)], dim=-1)
        
        v_final = slerp_vector(v_prior, v_pred, weight)
        
        # 3. 转回标量相位 p
        p_final = torch.atan2(v_final[..., 0:1], v_final[..., 1:2]) / (2 * torch.pi)
        p_final = p_final % 1.0

        # 4. 形态参数插值
        f_final = lerp(f_h, f_p, weight)
        a_final = lerp(a_h, a_p, weight)
        b_final = lerp(b_h, b_p, weight)

        return torch.cat([p_final, f_final, a_final, b_final], dim=1)

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")