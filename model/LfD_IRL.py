#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author  :   Fangze Lin
@System  :   Ubuntu20.04
@Time    :   2023/11/14 
@Brief   :   Learning Driving Styles for Autonomous Vehicles from Demonstration
'''

import torch
from model.NN_modules import *
import numpy as np
import os
from utils.test_utils import *
import theseus as th
from utils.train_utils import project_to_frenet_frame
from torch.utils.data import Dataset
from utils.polynomial import *

class MotionPlanner:
    def __init__(self, trajectory_len, feature_len, device, max_iterations=2, step_size=0.4, test=False):
        self.device = device

        # define cost function
        cost_function_weights = [th.ScaleCostWeight(th.Variable(torch.rand(1), name=f'cost_function_weight_{i+1}')) for i in range(feature_len)]

        # # define point weights
        # point_weights = [th.Vector(dof=50, name=f"point_weights_{i}") for i in range(feature_len)]

        # define control variable
        control_variables = th.Vector(dof=8, name="control_variables")
        
        # define prediction variable
        predictions = th.Variable(torch.empty(1, 10, trajectory_len, 3), name="predictions")

        # define ref_line_info
        ref_line_info = th.Variable(torch.empty(1, 600, 5), name="ref_line_info")
        
        # define current state
        current_state = th.Variable(torch.empty(1, 11, 8), name="current_state")

        # set up objective
        objective = th.Objective()
        self.objective = cost_function(objective, control_variables, current_state, predictions, ref_line_info, cost_function_weights)

        # set up optimizer
        if test:
            self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)
            # self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)
        else:
            self.optimizer = th.GaussNewton(objective, th.LUDenseSolver, vectorize=False, max_iterations=max_iterations, step_size=step_size, abs_err_tolerance=1e-2)
            # self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)

        # set up motion planner
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False) 
        self.layer.to(device=device)


# cost functions
def acceleration(optim_vars, aux_vars):
    control_variables = optim_vars[0].tensor.view(-1, 8)
    lat_a = control_variables[:, :4]
    lon_a = control_variables[:, 4:]

    lat_acc = cal_quintic_spline_second_derivative(lat_a)
    lon_acc = cal_quintic_spline_second_derivative(lon_a)

    acc = lat_acc + lon_acc
    # print("acc: ",acc.shape)
    return acc

def jerk(optim_vars, aux_vars):
    # control = optim_vars[0].tensor.view(-1, 50, 2)
    # acc = control[:, :, 0]
    control_variables = optim_vars[0].tensor.view(-1, 8)
    lat_a = control_variables[:, :4]
    lon_a = control_variables[:, 4:]

    lat_jerk = cal_quintic_spline_third_derivative(lat_a)
    lon_jerk = cal_quintic_spline_third_derivative(lon_a)
    jerk = lon_jerk + lat_jerk
    # print("jerk: ",jerk.shape)
    return jerk

# def steering(optim_vars, aux_vars):
#     control = optim_vars[0].tensor.view(-1, 50, 2)
#     steering = control[:, :, 1]

#     return steering 

# def steering_change(optim_vars, aux_vars):
#     control = optim_vars[0].tensor.view(-1, 50, 2)
#     steering = control[:, :, 1]
#     steering_change = torch.diff(steering) / 0.1

#     return steering_change

def speed(optim_vars, aux_vars):
    control_variables = optim_vars[0].tensor.view(-1, 8)
    lat_a = control_variables[:, :4]
    lon_a = control_variables[:, 4:]
    current_state = aux_vars[1].tensor[:, 0]

    lat_v = cal_quintic_spline_first_derivative(lat_a, current_state[:,3])
    lon_v = cal_quintic_spline_first_derivative(lon_a, current_state[:,4])
    # control = optim_vars[0].tensor.view(-1, 50, 2)
    # current_state = aux_vars[1].tensor[:, 0]
    # velocity = torch.hypot(current_state[:, 3], current_state[:, 4])
    # dt = 0.1

    speed = torch.hypot(lat_v, lon_v) # + velocity # 给定三角形两边，返回斜边
    # acc = control[:, :, 0]
    # speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(aux_vars[0].tensor[:, :, -1], dim=-1, keepdim=True)[0]
    speed_error = speed - speed_limit
    # print("speed_error: ",speed_error.shape)
    return speed_error

def lane_xy(optim_vars, aux_vars):
    # control = optim_vars[0].tensor.view(-1, 50, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]

    control_variables = optim_vars[0].tensor.view(-1, 8)
    lat_a = control_variables[:, :4]
    lon_a = control_variables[:, 4:]
    
    lat_loc = cal_quintic_spline_point(lat_a, current_state[:,0], current_state[:,3])
    lon_loc = cal_quintic_spline_point(lon_a, current_state[:,1], current_state[:,4])
    
    # traj = bicycle_model(control, current_state)
    traj = torch.stack([lat_loc, lon_loc],dim=-1)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0], traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)
    # print("lane_error: ",lane_error.shape)
    return lane_error

# def lane_theta(optim_vars, aux_vars):
#     control = optim_vars[0].tensor.view(-1, 50, 2)
#     ref_line = aux_vars[0].tensor
#     current_state = aux_vars[1].tensor[:, 0]
#     x_0, y_0 = current_state[:, 0], current_state[:, 1]
    

#     traj = bicycle_model(control, current_state)
#     distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
#     k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
#     ref_points = torch.gather(ref_line, 1, k)
#     theta = traj[:, :, 2]
#     lane_error = theta[:, 1::2] - ref_points[:, 1::2, 2]
    
#     return lane_error

# def red_light_violation(optim_vars, aux_vars):
#     control = optim_vars[0].tensor.view(-1, 50, 2)
#     current_state = aux_vars[1].tensor[:, 0]
#     ref_line = aux_vars[0].tensor
#     red_light = ref_line[..., -1]
#     dt = 0.1

#     velocity = torch.hypot(current_state[:, 3], current_state[:, 4])
#     acc = control[:, :, 0]
#     speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
#     speed = torch.clamp(speed, min=0)
#     s = torch.cumsum(speed * dt, dim=-1)

#     stop_point = torch.max(red_light[:, 200:]==0, dim=-1)[1] * 0.1
#     stop_distance = stop_point.view(-1, 1) - 3
#     red_light_error = (s - stop_distance) * (s > stop_distance) * (stop_point.unsqueeze(-1) != 0)

#     return red_light_error

def safety(optim_vars, aux_vars):
    control_variables = optim_vars[0].tensor.view(-1, 8)
    neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3)
    current_state = aux_vars[1].tensor
    ref_line = aux_vars[2].tensor
    lat_a = control_variables[:, :4]
    lon_a = control_variables[:, 4:]

    lat_loc = cal_quintic_spline_point(lat_a, current_state[:,0 ,0], current_state[:,0 ,3])
    lon_loc = cal_quintic_spline_point(lon_a, current_state[:,0 ,1], current_state[:,0 ,4])

    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    traj = torch.stack([lat_loc, lon_loc],dim=-1)
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]

    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(traj.detach(), ref_line)
    
    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]: # key frames
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
        safe_error.append(error)

    safe_error = torch.stack(safe_error, dim=1) 
    # print("safe_error: ",safe_error.shape)
    return safe_error

# TODO 基于采样的轨迹规划
def end_point(optim_vars, aux_vars):
    ...

def cost_function(objective, control_variables, current_state, predictions, ref_line, cost_function_weights, vectorize=True):
    # travel efficiency
    speed_cost = th.AutoDiffCostFunction([control_variables], speed, 50, cost_function_weights[0], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="speed")
    objective.add(speed_cost)

    # comfort
    acc_cost = th.AutoDiffCostFunction([control_variables], acceleration, 50, cost_function_weights[1], autograd_vectorize=vectorize, name="acceleration")
    objective.add(acc_cost)
    jerk_cost = th.AutoDiffCostFunction([control_variables], jerk, 50, cost_function_weights[2], autograd_vectorize=vectorize, name="jerk")
    objective.add(jerk_cost)
    # steering_cost = th.AutoDiffCostFunction([control_variables], steering, 50, cost_function_weights[3], autograd_vectorize=vectorize, name="steering")
    # objective.add(steering_cost)
    # steering_change_cost = th.AutoDiffCostFunction([control_variables], steering_change, 49, cost_function_weights[4], autograd_vectorize=vectorize, name="steering_change")
    # objective.add(steering_change_cost)
    
    # lane departure
    lane_xy_cost = th.AutoDiffCostFunction([control_variables], lane_xy, 50, cost_function_weights[3], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="lane_xy")
    objective.add(lane_xy_cost)
    # lane_theta_cost = th.AutoDiffCostFunction([control_variables], lane_theta, 25, cost_function_weights[6], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="lane_theta")
    # objective.add(lane_theta_cost)

    # traffic rules
    # red_light_cost = th.AutoDiffCostFunction([control_variables], red_light_violation, 50, cost_function_weights[8], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="red_light")
    # objective.add(red_light_cost)
    safety_cost = th.AutoDiffCostFunction([control_variables], safety, 10, cost_function_weights[4], aux_vars=[predictions, current_state, ref_line], autograd_vectorize=vectorize, name="safety")
    objective.add(safety_cost)

    return objective

if __name__ == "__main__":
    pass
    # set up model
    # model = SPPChannel(50)
    # print(model)
    # print('Model Params:', sum(p.numel() for p in model.parameters()))
