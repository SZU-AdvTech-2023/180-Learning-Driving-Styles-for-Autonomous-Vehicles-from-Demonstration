#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   Fangze Lin
@System  :   Ubuntu20.04
@Time    :   2023/11/14 
@Brief   :   Base Model
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from utils.test_utils import *
import theseus as th
from utils.train_utils import project_to_frenet_frame
from torch.utils.data import Dataset
from utils.polynomial import *

# Traj history encoder
class TrajEncoder(nn.Module):
    def __init__(self):
        super(TrajEncoder, self).__init__()
        self.motion = nn.LSTM(5, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output
    
# [轨迹类型*2，风格类型*3] 
class TrajPolicy(nn.Module):
    def __init__(self): # 0.1s / step
        super(TrajPolicy, self).__init__()
        self.traj_encoder = TrajEncoder()
        # control_variables_param
        # self.policy = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(64, 64),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(128, 7)
        # )
        # self.policy = nn.Sequential( # 2Tanh结构 输出：都为正
        #     nn.Linear(256, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 8)
        # )
        self.policy_x = nn.Sequential( # 2Tanh结构 输出：都为正
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.Linear(256, 256),
            # nn.Tanh(),
            nn.Linear(128, 16)
        )
        self.policy_y = nn.Sequential( # 2Tanh结构 输出：有正有负
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.Linear(256, 256),
            # nn.Tanh(),
            nn.Linear(128, 16)
        )
        self._parameters_init(True)
        # self.lat_policy = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4))
        # self.lon_policy = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 3))
        # self.register_buffer('control_variables_scale', torch.tensor([1, 1, 1, 1, 1])) # , 10, 100

    def _parameters_init(self, if_orthogonal):
        for n, m in self.policy_x.named_modules():
            # 这里仅初始化最后一层Linear的参数
            if n == "2" and isinstance(m, nn.Linear):
                print("parameters init layer: ", m)
                if if_orthogonal:
                    # 采用正交初始化
                    nn.init.orthogonal_(m.weight, 0.1)
                else:
                    m.weight.data.mul_(0.1)
                nn.init.constant_(m.bias, 0.0)

        for n, m in self.policy_y.named_modules():
            # 这里仅初始化最后一层Linear的参数
            if n == "2" and isinstance(m, nn.Linear):
                print("parameters init layer: ", m)
                if if_orthogonal:
                    # 采用正交初始化
                    nn.init.orthogonal_(m.weight, 0.1)
                else:
                    m.weight.data.mul_(0.1)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, batch_trajs): 
        # control_variables
        # print("control_var: ", batch_features.shape)
        # control_var = torch.concat([self.lat_policy(batch_features) ,self.lon_policy(batch_features)],dim=1)
        batch_features = self.traj_encoder(batch_trajs)
        control_var_x = self.policy_x(batch_features) 
        control_var_y = self.policy_y(batch_features) 
        control_var = torch.concatenate((control_var_x,control_var_y),dim=1)
        # print("control_var: ", control_var.shape)
        return control_var 
    
    def gaussian_sampling(self, batch_trajs, std = 1):
        control_var = self.forward(batch_trajs) 
        std = torch.ones_like(control_var) * std
        gaussian_control_var = torch.normal(control_var, std)
        return gaussian_control_var 
    
class BaseCostParamModel(nn.Module):
    def __init__(self, feature_len=5, point_len=50) -> None:
        super(BaseCostParamModel, self).__init__()
        self.feature_len = feature_len
        self.featuresNet = nn.Sequential(nn.Linear(1, feature_len), nn.Softmax(dim=-1))

    def forward(self, batch_dim, Param=None):
        if Param != None:
            """Param details:
            torch.Size([5, 1])
            torch.Size([5])
            """
            device = self.featuresNet[0].weight.device 
            """featuresNet"""
            f_input_ones = torch.ones(1, 1).to(device) 
            # Linear
            w, b = Param[0], Param[1]
            x = F.linear(f_input_ones, w, b) 
            cost_features_weights = F.softmax(x,dim=-1) # * self.features_scale
            return cost_features_weights
        else:
            # cost function weights 
            device = self.featuresNet[0].weight.device 
            f_input_ones = torch.ones(1, 1).to(device) 
            cost_features_weights = self.featuresNet(f_input_ones) # * self.features_scale
            return cost_features_weights

def cal_traj_features(traj, ref_line, prediction, current_state): # 先横再纵：十字顺序
    """base element"""
    lat_speed, lon_speed = torch.diff(traj[:, :, 0]) / 0.1, torch.diff(traj[:, :, 1]) / 0.1 # dim 49
    lat_speed, lon_speed = torch.clamp(lat_speed, min=-10., max=50.), torch.clamp(lon_speed, min=-20., max=40.)
    speed = torch.hypot(lat_speed, lon_speed) 
    """acc"""
    lat_acc, lon_acc = torch.diff(lat_speed) / 0.1, torch.diff(lon_speed) / 0.1, # dim 48
    f_lat_acc, f_lon_acc = torch.clamp(lat_acc[:, :47], min=-200., max=200.), torch.clamp(lon_acc[:, :47], min=-200., max=200.)
    """jerk"""
    lat_jerk, lon_jerk = torch.diff(lat_speed, n=2) / 0.01, torch.diff(lon_speed, n=2) / 0.01 # dim 47
    f_lat_jerk, f_lon_jerk = torch.clamp(lat_jerk[:, :47], min=-4000., max=2000.), torch.clamp(lon_jerk[:, :47], min=-4000., max=2000.)
    """curvature"""
    # print((f_lon_acc * lat_speed[:, :47] - f_lat_acc * lon_speed[:, :47]).mean(dim=1).data)
    # print(((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2)).mean(dim=1).data)
    f_k_up = (f_lon_acc * lat_speed[:, :47] - f_lat_acc * lon_speed[:, :47])* 0.01
    f_k_down =  ((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2))*0.0001
    # print(f_k.mean(dim=1).data)
    f_k = f_k_up / (f_k_down + 1e-3)
    f_k2 = f_k**2  # dim 47
    """speed efficiency"""
    speed_limit = torch.max(ref_line[:, :, -1], dim=-1, keepdim=True)[0]
    abs_speed_limit = torch.abs(speed_limit - speed[:,:47])
    f_efficiency = torch.mean(abs_speed_limit, dim=1) # dim 1
    # f_efficiency2 = speed[:,:47] - speed_limit
    """lane"""
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    # f_lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0], traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)
    f_lane_error = torch.hypot(traj[:, :47, 0]-ref_points[:, :47, 0], traj[:, :47, 1]-ref_points[:, :47, 1]) # dim47
    """collision avoidance"""
    neighbors = prediction.permute(0, 2, 1, 3)
    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]
    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(traj.detach(), ref_line)

    safe_error = []
    for t in range(47): # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :, 0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(traj[:, t, :2].unsqueeze(1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error) # dim 47

    f_safe_error = torch.stack(safe_error, dim=1)

    # features
    features = torch.stack([
                torch.sum((0.01*f_lat_acc),dim=1), 
                torch.sum((0.01*f_lon_acc),dim=1), 
                torch.sum((0.001*f_lat_jerk),dim=1), 
                torch.sum((0.01*f_lon_jerk),dim=1), 
                torch.sum((0.0001*f_k2),dim=1), 
                (0.03*f_efficiency), 
                torch.sum((0.0002*f_lane_error),dim=1), 
                torch.sum((0.002*f_safe_error),dim=1)
                ],dim=1)
    
    # print("features: ", features.shape)
    # print("E_features: ", E_features.shape)
    return features

if __name__ == "__main__":
    pass
    # set up model
    # model = SPPChannel(50)
    # print(model)
    # print('Model Params:', sum(p.numel() for p in model.parameters()))
