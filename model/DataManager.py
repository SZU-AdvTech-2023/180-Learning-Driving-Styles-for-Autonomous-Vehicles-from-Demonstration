#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SPP.py
@Author  :   Fangze Lin
@System  :   Ubuntu20.04
@Time    :   2023/07/14 16:05:15
@Brief   :   Spp: channel 1.95% struct1
1、single traj
2、late fusion
'''

import torch
from model.NN_modules import *
import numpy as np
import os
from utils.test_utils import *
import theseus as th
from utils.train_utils import project_to_frenet_frame
from torch.utils.data import Dataset
import random

def msirl_metrics(plan_trajectory, ground_truth_trajectories):
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)
    # planning
    plannerADE1 = torch.mean(plan_distance[:, :10])
    plannerADE2 = torch.mean(plan_distance[:,:30])
    plannerADE3 = torch.mean(plan_distance[:,:50])
    plannerFDE1 = torch.mean(plan_distance[:, 9])
    plannerFDE2 = torch.mean(plan_distance[:, 29])
    plannerFDE3 = torch.mean(plan_distance[:, 49])
    return plannerADE1.item(),plannerADE2.item(),plannerADE3.item(),plannerFDE1.item(),plannerFDE2.item(),plannerFDE3.item()

        
import glob

class DataManager_Run(Dataset):
    def __init__(self, running_dir, save_dir = None, is_rand = False):
        self.is_rand = is_rand
        self.running_dir = running_dir
        self.save_dir = save_dir
        if not os.path.exists(self.running_dir):
            os.mkdir(self.running_dir)
        if save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir) #递归创建目录
        self.data_list = glob.glob(self.running_dir+'/*')

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx): 
        path_list = self.data_list[idx].split('/')[-1].split('_')
        behavior_i, style_i = int(path_list[0].split('behavior')[-1]), int(path_list[1].split('style')[-1].split('.')[0])

        if self.is_rand:
            style_i = random.randint(0, 4)
        return self.load_data(behavior_i, style_i)
    
    def _check_nan_(self, *data):
        is_nan = False
        for elem in data:
            if np.isnan(elem).any():
                is_nan = True
        return is_nan
    
    def load_data(self, behavior_i, style_i): # load traj
        filename = f"{self.running_dir}/behavior{behavior_i}_style{style_i}.npz"
        # load data
        data = np.load(filename)
        # prediction = data['prediction']
        plan_traj = data['plan_traj']
        ref_line = data['ref_line']
        current_state = data['current_state']
        ground_truth = data['ground_truth']
        return ref_line, current_state, ground_truth, behavior_i, style_i, plan_traj
    
    # def save_data(self, behavior_i, style_i, weights): # save task
    #     if self.save_dir is not None:
    #         if self._check_nan_(weights):
    #             return 
    #         filename = f"{self.save_dir}/behavior{behavior_i}_style{style_i}_weights.npz"
    #         # save data
    #         np.save(filename, weights)
    #     else:
    #         raise Exception("self.save_dir is None ! ! !")
        


class DataManager_Task(Dataset):
    def __init__(self, running_dir,save_dir):
        self.running_dir = running_dir
        self.save_dir = save_dir
        if not os.path.exists(self.running_dir):
            os.mkdir(self.running_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir) #递归创建目录
        self.data_list = glob.glob(self.running_dir+'/*')

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx): 
        path_list = self.data_list[idx].split('/')[-1].split('_')
        scene_id, time_step = path_list[0], int(path_list[1].split('.')[0])

        return self.load_data(scene_id, time_step)
    
    def _check_nan_(self, *data):
        is_nan = False
        for elem in data:
            if np.isnan(elem).any():
                is_nan = True
        return is_nan
    
    def load_data(self, scene_id, time_step): # load traj
        filename = f"{self.running_dir}/{scene_id}_{time_step}.npz"
        # load data
        data = np.load(filename)
        ego = data['ego']
        # prediction = data['prediction']
        plan = data['plan']
        ref_line = data['ref_line']
        current_state = data['current_state']
        ground_truth = data['ground_truth']
        return ref_line, current_state, ground_truth, ego, plan # prediction, plan, 
    
    def save_data(self, behavior_i, style_i, plan_traj, ref_line, current_state, ground_truth): # save task
        if self._check_nan_(plan_traj, ref_line, current_state, ground_truth):
            return 
        filename = f"{self.save_dir}/behavior{behavior_i}_style{style_i}.npz"
        # save data
        np.savez(filename, plan_traj=plan_traj, ref_line=ref_line, current_state=current_state,ground_truth=ground_truth)


class DataManager_Traj(Dataset):
    def __init__(self, running_dir,save_dir):
        self.running_dir = running_dir
        self.save_dir = save_dir
        if not os.path.exists(self.running_dir):
            os.mkdir(self.running_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir) #递归创建目录
        self.data_list = glob.glob(self.running_dir+'/*')

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx): # load raw
        path_list = self.data_list[idx].split('/')[-1].split('_')
        scene_id, time_step = path_list[0], int(path_list[1].split('.')[0])
        
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line']
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']
        
        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states, scene_id, time_step
    
    def _check_nan_(self, *data):
        is_nan = False
        for elem in data:
            if np.isnan(elem).any():
                is_nan = True
        return is_nan

    def save_data(self, scene_id, time_step, ego, prediction, plan, ref_line, current_state, ground_truth): # save traj
        if self._check_nan_(prediction, plan, ref_line, current_state, ground_truth):
            return 
        # filename = f"{self.running_dir}/{scenario_id}.npz"
        # np.savez(filename, prediction=prediction, plan=plan, ref_line=ref_line, current_state=current_state,ground_truth=ground_truth)

        # for i in range(time_step_list.shape[0]):
        #     filename = f"{self.running_dir}/{scenario_id}_{time_step[i]}.npz"
        #     # save data
        #     np.savez(filename, prediction=prediction, plan=plan, ref_line=ref_line, current_state=current_state,ground_truth=ground_truth,comfort_cost=comfort_cost,efficiency_cost=efficiency_cost)
        
        for i,scene_id in enumerate(scene_id):
            filename = f"{self.save_dir}/{scene_id}_{time_step[i]}.npz"
            # save data
            np.savez(filename, ego=ego[i], prediction=prediction[i], plan=plan[i], ref_line=ref_line[i], current_state=current_state[i],ground_truth=ground_truth[i])


if __name__ == "__main__":
    pass
    # set up model
    # model = SPPChannel(50)
    # print(model)
    # print('Model Params:', sum(p.numel() for p in model.parameters()))
