
import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import *
from utils.test_utils import show_figure
from torch.utils.data import DataLoader
from model.DataManager import *
from model.LfD_Base import *
import copy
from model.LfD_IRL import MotionPlanner

# 设置打印选项
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.3f}'.format})

def show_metrics(metrics,str="raw"):
    epoch_metrics = np.array(metrics)
    plannerADE1, plannerFDE1 = epoch_metrics[0], epoch_metrics[3]
    plannerADE2, plannerFDE2 = epoch_metrics[1], epoch_metrics[4]
    plannerADE3, plannerFDE3 = epoch_metrics[2], epoch_metrics[5]
    epoch_metrics = [plannerADE1, plannerADE2, plannerADE3, plannerFDE1, plannerFDE2, plannerFDE3]
    logging.info(str + f'-plannerADE1: {plannerADE1:.4f}, ' + str + f'-plannerFDE1: {plannerFDE1:.4f}')
    logging.info(str + f'-plannerADE1: {plannerADE2:.4f}, ' + str + f'-plannerFDE1: {plannerFDE2:.4f}')
    logging.info(str + f'-plannerADE1: {plannerADE3:.4f}, ' + str + f'-plannerFDE1: {plannerFDE3:.4f}')
    return epoch_metrics

def show_batch_metrics(metrics,str="raw"):
    epoch_metrics = np.array(metrics)
    plannerADE1, plannerFDE1 = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 3])
    plannerADE2, plannerFDE2 = np.mean(epoch_metrics[:, 1]), np.mean(epoch_metrics[:, 4])
    plannerADE3, plannerFDE3 = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [plannerADE1, plannerADE2, plannerADE3, plannerFDE1, plannerFDE2, plannerFDE3]
    logging.info(str + f'-plannerADE1: {plannerADE1:.4f}, ' + str + f'-plannerFDE1: {plannerFDE1:.4f}')
    logging.info(str + f'-plannerADE1: {plannerADE2:.4f}, ' + str + f'-plannerFDE1: {plannerFDE2:.4f}')
    logging.info(str + f'-plannerADE1: {plannerADE3:.4f}, ' + str + f'-plannerFDE1: {plannerFDE3:.4f}')
    return epoch_metrics

def check_stop(loss_val,before_loss_val,raw_loss_val,stop_value=-1):
    if stop_value <= 0:
        stop_value = 1e-3
    if loss_val< args.stop_threshold: # soft: [3e-4,1e-3] min(max(args.stop_threshold,raw_loss_val/2),1e-3)
        return True
    # elif loss_val < min(stop_value*1.2,raw_loss_val*0.1): # plus
    #     return True
    elif loss_val < min(before_loss_val/4,stop_value*1.2,raw_loss_val*0.3): # raw
        return True
    elif loss_val < min(before_loss_val/8,stop_value*1.2,raw_loss_val*0.6):
        return True
    elif loss_val < min(before_loss_val/12,stop_value*1.2,raw_loss_val*0.9):
        return True
    elif loss_val < min(before_loss_val/32,stop_value*1.2): # plus
        return True
    return False 
    # if loss_val< args.stop_threshold: # soft: [3e-4,1e-3] min(max(args.stop_threshold,raw_loss_val/2),1e-3)
    #     return True
    # elif loss_val < min(before_loss_val/6,stop_value*1.2,raw_loss_val/3): # raw
    #     return True
    # elif loss_val < min(before_loss_val/12,stop_value*1.2,raw_loss_val):
    #     return True
    # elif loss_val < min(before_loss_val/32,stop_value*1.2): # plus
    #     return True
    # return False 

def LfD_Loss(paramModel, planner, E_expert_features,control_var,ground_truth,ref_line,current_state,is_opt=True):
    if is_opt:
        cost_features_weights = paramModel(batch_dim = current_state.shape[0])
        # IRL
        planner_inputs = {
            "control_variables": control_var.detach().view(-1, 8),
            "predictions": ground_truth[:,1:,:,:3],
            "ref_line_info": ref_line,
            "current_state": current_state
        }

        for i in range(cost_features_weights.shape[1]):
            planner_inputs[f'cost_function_weight_{i+1}'] = cost_features_weights[:, i].unsqueeze(1) 

        final_values, info = planner.layer.forward(planner_inputs) # , optimizer_kwargs={'track_best_solution': True}
        # control_var = info.best_solution['control_variables'].view(-1, 8).to(device)
        # final_values, info = planner.layer.forward(planner_inputs)
        control_var = final_values["control_variables"].view(-1, 8)

    # print("opt_control_var: ", control_var[0])
    lat_loc = cal_quintic_spline_point(control_var[:,:4], current_state[:, 0 ,0], current_state[:, 0 ,3])
    lon_loc = cal_quintic_spline_point(control_var[:,4:], current_state[:, 0 ,1], current_state[:, 0 ,4])
    traj = torch.stack([lat_loc, lon_loc], dim=-1)

    # IRL objective
    E_learner_features = cal_traj_features(traj, ref_line, ground_truth[:,1:], current_state).mean(dim=0)

    # print(E_learner_features.data)
    # print(E_expert_features.data)
    # check
    if torch.isnan(E_learner_features).any().item() or torch.isnan(E_expert_features).any().item():
        raise Exception("There are Nan in IRL_loss !!!")
    plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
    IRL_loss = F.smooth_l1_loss(E_learner_features, E_expert_features)
    traj_loss = F.smooth_l1_loss(traj, ground_truth[:, 0, :, :2]) 
    end_loss = F.smooth_l1_loss(traj[:, -1], ground_truth[:, 0, -1, :2])
    loss = IRL_loss # + traj_loss + end_loss # + 1e-3 * plan_cost
    info_str = f"loss:{loss.item():.4f}  IRL-loss:{IRL_loss.item():.4f}  end-loss:{end_loss.item():.4f}  traj-loss:{traj_loss.item():.4f}  plan-cost:{plan_cost.item():.4f}"
    return loss, traj, info_str, control_var

def train_task():
    # device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    # Logging
    log_name = __file__.split('/')[-1].split('.')[0] + f"_{args.cost_type}"
    log_path = f"./training_log/{log_name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+log_name+'_train.log')

    # set seed
    fixed_seed(args.seed)
    trajectory_len, feature_len = 50, 5
    max_iterations, opt_step_size= 1, 0.3
    planner = MotionPlanner(trajectory_len, feature_len, device, max_iterations, opt_step_size)
    t_planner = MotionPlanner(trajectory_len, feature_len, device, max_iterations=args.step_N, step_size=opt_step_size)
    
    logging.info("------------- {} -------------".format(log_name))
    logging.info("seed: {}".format(args.seed))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(device))
    # logging.info("scheduler_step_size: {}".format(args.scheduler_step_size))
    logging.info("learn_epoch_max: {}".format(args.learn_epoch_max))
    logging.info("step_N: {}".format(args.step_N))
    logging.info("max_iterations: {}".format(max_iterations))
    logging.info("opt_step_size: {}".format(opt_step_size))

    # set up data loaders
    # train_manager = DataManager_Run(args.train_path)
    test_manager = DataManager_Run(args.test_path)
    # train_loader = DataLoader(train_manager, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_manager, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # logging.info("Dataset Prepared: {} train data\n".format(len(train_manager)))
    logging.info("Dataset Prepared: {} test data\n".format(len(test_manager)))

    # train_size = len(train_loader.dataset)
    start_time = time.time()
    traj_policy = TrajPolicy().to(device)
    traj_policy.load_state_dict(torch.load(args.policy_path, map_location=device))
    traj_policy.eval()
    # prediction, plan, ref_line, current_state, ground_truth
    for task_epoch in range(1, args.task_epochs+1):
        # logging.info(f"\n==========================task: {task_epoch}============================")
        for task_i, x_spt in enumerate(test_loader):
            logging.info(f"==========================task_i: {task_i+1}============================")
            """train"""
            # prepare data
            # prediction = x_spt[0][0].to(device)
            ref_line = x_spt[0][0].to(device)
            current_state = x_spt[1][0].to(device)
            ground_truth = x_spt[2][0].to(device)
            behavior_i = x_spt[3][0].item()
            style_i = x_spt[4][0].item()
            logging.info(f"behavior_i: {behavior_i}  style_i: {style_i}")
            # plan = x_spt[5][0].to(device)

            
            # mean_ground_truth = torch.mean(ground_truth[:, 0,:,:],dim=0).unsqueeze(0)

            # mean_ground_truth = torch.min(ground_truth[:, 0,:,:],dim=0)[0]
            # print("prediction",prediction.shape)
            # print("plan",plan.shape)
            # print("ref_line",ref_line.shape) 
            # print("current_state",current_state.shape)
            # print("ground_truth",ground_truth.shape)
            # print("comfort_cost",comfort_cost.shape)
            # print("efficiency_cost",efficiency_cost.shape)
            """
            prediction torch.Size([64, 10, 50, 3])
            plan torch.Size([64, 50, 2])
            ref_line torch.Size([64, 1200, 5])
            current_state torch.Size([64, 11, 8])
            ground_truth torch.Size([64, 11, 50, 5])
            comfort_cost torch.Size([64])
            efficiency_cost torch.Size([64])
            """
            # train init
            paramModel = BaseCostParamModel(feature_len=5,point_len=50).to(device)
            optimizer = optim.Adam(paramModel.parameters(), lr = args.learning_rate)
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.scheduler_step_size, gamma=0.5)
            E_expert_features = cal_traj_features(ground_truth[:, 0],ref_line,ground_truth[:,1:],current_state).mean(dim=0)
            ###################################################################################################
            """traj generation: 生成N条init轨迹(check raw)""" 
            # generate check trajs
            control_var_random_check = [None]*(args.check_traj_N+1)
            control_var_before_check = [None]*(args.check_traj_N+1)
            raw_check_metrics = [None]*(args.check_traj_N+1)
            raw_check_loss_val = [None]*(args.check_traj_N+1)
            raw_check_info_str = [None]*(args.check_traj_N+1)
            with torch.no_grad():
                for i in range(args.check_traj_N+1):
                    control_var_random_check[i] = traj_policy.gaussian_sampling(ground_truth[:, 0], std=0.002)
                    control_var_before_check[i] = copy.deepcopy(control_var_random_check[i])
                    raw_loss, raw_traj, raw_check_info_str[i], _ = LfD_Loss(paramModel,planner,E_expert_features,control_var_random_check[i],ground_truth,ref_line,current_state,is_opt=False)
                    raw_check_loss_val[i] = raw_loss.item()
                    raw_check_metrics[i] = msirl_metrics(raw_traj, ground_truth)
            ###################################################################################################
            """raw"""
            # select learned traj
            learn_traj_idx = np.argmax(np.array(raw_check_loss_val))
            control_var = control_var_random_check[learn_traj_idx]   
            control_var_before = control_var_before_check[learn_traj_idx]
            control_var_after = copy.deepcopy(control_var)
            raw_loss_val = raw_check_loss_val[learn_traj_idx]                
            raw_info_str = raw_check_info_str[learn_traj_idx]
            raw_metrics = raw_check_metrics[learn_traj_idx]
            # del selected traj
            del control_var_random_check[learn_traj_idx],control_var_before_check[learn_traj_idx]
            del raw_check_loss_val[learn_traj_idx],raw_check_info_str[learn_traj_idx],raw_check_metrics[learn_traj_idx]
            logging.info(f"learn result:")
            logging.info(f"raw loss--> "+raw_info_str)
            """before learn"""
            opt_start_time = time.time()
            with torch.no_grad():
                for before_step in range(1,args.step_N+1):
                    before_loss, before_traj, info_str, control_var_before = LfD_Loss(paramModel,planner,E_expert_features,control_var_before,ground_truth,ref_line,current_state)
                    # if before_loss.item()< args.stop_threshold:
                    #         break
            before_time = time.time() - opt_start_time
            before_metrics = msirl_metrics(before_traj, ground_truth)
            logging.info(f"before learn --> "+info_str)
            before_loss_val = before_loss.item()
            """check before"""
            before_check_step = [None] * args.check_traj_N
            before_check_metrics = [None] * args.check_traj_N
            before_check_info_str = [None] * args.check_traj_N
            before_check_loss_val = [None] * args.check_traj_N
            before_check_time = [None] * args.check_traj_N
            for i in range(args.check_traj_N):
                opt_start_time = time.time()
                with torch.no_grad():
                    for before_check_i in range(1,args.step_N+1):
                        before_check_loss, before_check_traj,  before_check_info_str[i], control_var_before_check[i] = LfD_Loss(paramModel,planner,E_expert_features,control_var_before_check[i],ground_truth,ref_line,current_state)
                        if check_stop(before_check_loss.item(),before_loss_val,raw_loss_val): # (learn_step>3 or learn_step>before_step-1) and 
                            break
                    before_check_time[i] = time.time() - opt_start_time
                    before_check_step[i] = before_check_i
                    before_check_metrics[i] = msirl_metrics(before_check_traj, ground_truth)
                    before_check_loss_val[i] = before_check_loss.item()
            # before_check_loss_val = before_check_loss.item()
            ###################################################################################################
            """learn process"""
            opt_start_time = time.time()
            suc_flag = False
            learn_step = 0
            stop_value = -1
            min_value = 100
            for learn_epoch in range(args.learn_epoch_max):
                # with torch.no_grad():
                #     if epoch==args.train_epochs-1:
                #         control_var = control_var_after
                #     else:
                #         control_var = traj_policy.gaussian_sampling(ground_truth[:, 0], std=0.001)
                # learn episode
                if args.cost_type == "lfd":
                    learn_step = 20
                    learn_loss, cross_traj, step_info_str, _ = LfD_Loss(paramModel,t_planner,E_expert_features,control_var,ground_truth,ref_line,current_state)
                    print(f"\rEpoch: {learn_epoch} " + step_info_str, end=" ") 
                    # update stop value
                    if learn_loss.item() < min_value:
                        min_value = learn_loss.item()
                    # stop condition
                    if learn_epoch*args.step_N+learn_step>10 and check_stop(learn_loss.item(),before_loss_val,raw_loss_val,stop_value=stop_value): # 
                        break 
                    # update IRL
                    # loss backward
                    optimizer.zero_grad() 
                    learn_loss.backward() # retain_graph=True 
                    optimizer.step() 
                    learn_loss_val = learn_loss.item()
                    stop_value = min_value
                else:
                    for learn_step in range(1,args.step_N+1):
                        learn_loss, cross_traj, step_info_str, control_var = LfD_Loss(paramModel,planner,E_expert_features,control_var,ground_truth,ref_line,current_state)
                        print(f"\rEpoch: {learn_epoch} Train-step: {learn_step}/{args.step_N} " + step_info_str, end=" ") 
                        # update stop value
                        if learn_loss.item() < min_value:
                            min_value = learn_loss.item()
                        # stop condition
                        if learn_epoch*args.step_N+learn_step>10 and check_stop(learn_loss.item(),before_loss_val,raw_loss_val,stop_value=stop_value): # 
                            suc_flag = True
                            break 
                        # update IRL
                        # loss backward
                        optimizer.zero_grad() 
                        learn_loss.backward() # retain_graph=True 
                        optimizer.step() 
                    learn_loss_val = learn_loss.item()
                    if suc_flag:
                        break
                    stop_value = min_value
                    control_var = copy.deepcopy(control_var_after)
            learn_time = time.time() - opt_start_time 
            # scheduler.step() 
            cross_metrics = msirl_metrics(cross_traj, ground_truth)
            logging.info(f"\rcross learn --> " + step_info_str + f"  stop_value:{stop_value:>.4f}  ")
            ################################################################################################### 
            """after learn""" 
            opt_start_time = time.time() 
            with torch.no_grad(): # 节省计算图开销 
                for after_step in range(1,args.step_N+1):
                    after_loss, after_traj, after_info_str, control_var_after = LfD_Loss(paramModel,planner,E_expert_features,control_var_after,ground_truth,ref_line,current_state)
                    if check_stop(after_loss.item(), before_loss_val, raw_loss_val, stop_value): # (learn_step>3 or learn_step>before_step-1) and 
                        break
                after_loss_val = after_loss.item()
            test_time = time.time() - opt_start_time
            after_metrics = msirl_metrics(after_traj, ground_truth)
            logging.info(f"after learn --> " + after_info_str)
            ###################################################################################################
            """random check"""
            check_step = [None] * args.check_traj_N
            check_metrics = [None] * args.check_traj_N
            check_info_str = [None] * args.check_traj_N
            check_loss_val = [None] * args.check_traj_N
            check_time = [None] * args.check_traj_N 
            for i in range(args.check_traj_N):
                opt_start_time = time.time() 
                with torch.no_grad(): # 节省计算图开销 
                    for check_i in range(1,args.step_N+1):
                        check_loss, check_traj, check_info_str[i], control_var_random_check[i] = LfD_Loss(paramModel,planner,E_expert_features,control_var_random_check[i],ground_truth,ref_line,current_state)
                        if check_stop(check_loss.item(),before_loss_val,raw_loss_val,stop_value): # (learn_step>3 or learn_step>before_step-1) and 
                            break
                    check_time[i] = time.time() - opt_start_time
                    check_step[i] = check_i
                    check_metrics[i] = msirl_metrics(check_traj, ground_truth)
                    check_loss_val[i] = check_loss.item()
                    
            """check result"""
            # logging.info(f"check result:")
            # # loss val
            # logging.info(f"check raw loss val--> {np.round(raw_check_loss_val,4)}")
            # logging.info(f"check before loss val--> {np.round(before_check_loss_val,4)}")
            # logging.info(f"check after loss val--> {np.round(check_loss_val,4)}")
            # # run step
            # logging.info(f"check before run step--> {before_check_step}")
            # logging.info(f"check after run step--> {check_step}")
            ###################################################################################################
            learn_step_N = learn_epoch*args.step_N+learn_step
            print(f"\r|| Learn-step: {learn_step_N} ||") # t_planner的结果/与train-step结果略有不同

            # show train metrics
            logging.info("train metrics:")
            raw_epoch_metrics = show_metrics(raw_metrics,str="raw")
            before_epoch_metrics = show_metrics(before_metrics,str="before")
            cross_epoch_metrics = show_metrics(cross_metrics,str="cross")
            after_epoch_metrics = show_metrics(after_metrics,str="after")

            # show check metrics
            logging.info("check metrics:")
            raw_check_epoch_metrics = show_batch_metrics(raw_check_metrics,str="raw_check")
            before_check_epoch_metrics = show_batch_metrics(before_check_metrics,str="before_check")
            check_epoch_metrics = show_batch_metrics(check_metrics,str="check")
            # show time
            logging.info("In sum: ")
            logging.info("""******************learn*******************""")
            logging.info(f"""Task-Progress: [{task_i+1:>3d}/{len(test_manager):>3d}]  Time: {(time.time()-start_time)/(task_i+1):>.3f}s/task  """)
            logging.info(f"""before-Time:{before_time:>.3f}s|{before_step}-step  learn-Time: {learn_time:>.3f}s|{learn_step_N}-step  after-Time: {test_time:>.3f}s|{after_step}-step """)
            logging.info("""******************check*******************""")
            for i in range(args.check_traj_N):
                logging.info(f"""before-Time:{before_check_time[i]:>2.3f}s|{before_check_step[i]:2d}-step  check-Time:{check_time[i]:>2.3f}s|{check_step[i]:2d}-step  loss-Val[raw|before|after]:[{raw_check_loss_val[i]:>.4f}|{before_check_loss_val[i]:>.4f}|{check_loss_val[i]:>.4f}]""")
            logging.info(f"""check mean result:""")
            logging.info(f"""before-MeanTime:{np.mean(before_check_time):>.3f}s|{np.mean(before_check_step):>.1f}-step  check-MeanTime:{np.mean(check_time):>.3f}s|{np.mean(check_step):>.1f}-step  loss-MeanVal[raw|before|after]:[{np.mean(raw_check_loss_val):>.4f}|{np.mean(before_check_loss_val):>.4f}|{np.mean(check_loss_val):>.4f}]""")
            logging.info("""******************************************""")

            # save to training log
            log = {
                    # time
                    'task_i': task_i, 'learn_time': learn_time, 'before_time':before_time, 'test_time': test_time,
                    'before_check_time':before_check_time, 'check_time':check_time,'stop_value':stop_value,
                    # step
                    'before_step':before_step, 'learn_step_N':learn_step_N, 'after_step':after_step,
                    'before_check_step':before_check_step, 'check_step':check_step,
                    # loss
                    'raw_loss_val':raw_loss_val, 'before_loss_val':before_loss_val, 'learn_loss_val':learn_loss_val, 'after_loss_val':after_loss_val,
                    'raw_check_loss_val':raw_check_loss_val, 'before_check_loss_val':before_check_loss_val, 'check_loss_val': check_loss_val,
                    # metrics
                    'raw-plannerADE1': raw_epoch_metrics[0], 'raw-plannerFDE1': raw_epoch_metrics[3], 
                    'raw-plannerADE2': raw_epoch_metrics[1], 'raw-plannerFDE2': raw_epoch_metrics[4],
                    'raw-plannerADE3': raw_epoch_metrics[2], 'raw-plannerFDE3': raw_epoch_metrics[5],
                    'before-plannerADE1': before_epoch_metrics[0], 'before-plannerFDE1': before_epoch_metrics[3], 
                    'before-plannerADE2': before_epoch_metrics[1], 'before-plannerFDE2': before_epoch_metrics[4],
                    'before-plannerADE3': before_epoch_metrics[2], 'before-plannerFDE3': before_epoch_metrics[5],
                    'cross-plannerADE1': cross_epoch_metrics[0], 'cross-plannerFDE1': cross_epoch_metrics[3], 
                    'cross-plannerADE2': cross_epoch_metrics[1], 'cross-plannerFDE2': cross_epoch_metrics[4],
                    'cross-plannerADE3': cross_epoch_metrics[2], 'cross-plannerFDE3': cross_epoch_metrics[5],
                    'after-plannerADE1': after_epoch_metrics[0], 'after-plannerFDE1': after_epoch_metrics[3], 
                    'after-plannerADE2': after_epoch_metrics[1], 'after-plannerFDE2': after_epoch_metrics[4],
                    'after-plannerADE3': after_epoch_metrics[2], 'after-plannerFDE3': after_epoch_metrics[5],
                    # check metrics
                    'raw_check-plannerADE1': raw_check_epoch_metrics[0], 'raw_check-plannerFDE1': raw_check_epoch_metrics[3], 
                    'raw_check-plannerADE2': raw_check_epoch_metrics[1], 'raw_check-plannerFDE2': raw_check_epoch_metrics[4],
                    'raw_check-plannerADE3': raw_check_epoch_metrics[2], 'raw_check-plannerFDE3': raw_check_epoch_metrics[5],
                    'before_check-plannerADE1': before_check_epoch_metrics[0], 'before_check-plannerFDE1': before_check_epoch_metrics[3], 
                    'before_check-plannerADE2': before_check_epoch_metrics[1], 'before_check-plannerFDE2': before_check_epoch_metrics[4],
                    'before_check-plannerADE3': before_check_epoch_metrics[2], 'before_check-plannerFDE3': before_check_epoch_metrics[5],
                    'check-plannerADE1': check_epoch_metrics[0], 'check-plannerFDE1': check_epoch_metrics[3], 
                    'check-plannerADE2': check_epoch_metrics[1], 'check-plannerFDE2': check_epoch_metrics[4],
                    'check-plannerADE3': check_epoch_metrics[2], 'check-plannerFDE3': check_epoch_metrics[5]
                    }

            if task_i == 0:
                with open(f'./training_log/{log_name}/train_log.csv', 'w') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with open(f'./training_log/{log_name}/train_log.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())
            
            """save learned weights"""
            save_path = log_path + f"save_model/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(paramModel.state_dict(), save_path+f"behavior_{behavior_i}_style_{style_i}.pth")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='LfD_Training')
    parser.add_argument('--cost_type', type=str, help='type of cost function [lfd/raw/point/point_cut]', default='lfd')
    parser.add_argument('--test_path', type=str, help='path to validation datasets', default='/home/lin/waymo_dataset/task_test')
    parser.add_argument('--policy_path', type=str, help='path to saved model', default='/home/lin/waymo_dataset/model_param/policy100.pth')
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument('--num_workers', type=int, help="number of workers used for dataloader", default=8)
    parser.add_argument('--check_traj_N', type=int, help='initial trajectory for checking result', default=10)
    parser.add_argument('--step_N', type=int, help='epochs of task training', default=20)
    parser.add_argument('--task_epochs', type=int, help='epochs of task training', default=1)
    parser.add_argument('--learn_epoch_max', type=int, help='max epoch of learning', default=25) # 20
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=1e-2)
    parser.add_argument('--stop_threshold', type=float, help='stop opt threshold (default: 0.0003)', default=3e-4)
    parser.add_argument('--gpu_id', type=int, help='run on which device (default: cuda)', default=0)
    args = parser.parse_args()

    # Run
    train_task()
