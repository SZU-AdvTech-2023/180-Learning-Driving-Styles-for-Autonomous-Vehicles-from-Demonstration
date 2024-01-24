
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
from torch.utils.data import DataLoader
from model.DataManager import *
from model.LfD_Base import *
import copy
from model.LfD_IRL import MotionPlanner



# 设置打印选项
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.3f}'.format})

def show_metrics(metrics,str="raw"):
    epoch_metrics = np.array(metrics)
    plannerADE1, plannerFDE1 = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 3])
    plannerADE2, plannerFDE2 = np.mean(epoch_metrics[:, 1]), np.mean(epoch_metrics[:, 4])
    plannerADE3, plannerFDE3 = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [plannerADE1, plannerADE2, plannerADE3, plannerFDE1, plannerFDE2, plannerFDE3]
    logging.info(str + f'-plannerADE1: {plannerADE1:.4f}, ' + str + f'-plannerFDE1: {plannerFDE1:.4f}')
    logging.info(str + f'-plannerADE1: {plannerADE2:.4f}, ' + str + f'-plannerFDE1: {plannerFDE2:.4f}')
    logging.info(str + f'-plannerADE1: {plannerADE3:.4f}, ' + str + f'-plannerFDE1: {plannerFDE3:.4f}')
    return epoch_metrics

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
        # control_var = info.best_solution['control_variables'].view(-1, 7).to(device)
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
    loss = IRL_loss + traj_loss + end_loss  # + 1e-3 * plan_cost
    info_str = f"loss: {loss.item():.4f} IRL-loss: {IRL_loss.item():.4f}  end-loss: {end_loss.item():.4f}  traj-loss: {traj_loss.item():.4f}  plan-cost: {plan_cost.item():.4f}"
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
    # logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(device))
    # logging.info("scheduler_step_size: {}".format(args.scheduler_step_size))
    logging.info("train_epochs: {}".format(args.train_epochs))
    logging.info("step_N: {}".format(args.step_N))
    logging.info("max_iterations: {}".format(max_iterations))
    logging.info("opt_step_size: {}".format(opt_step_size))

    # set up data loaders
    train_manager = DataManager_Run(args.train_path)
    # after_manager = DataManager_Run(args.after_path)
    train_loader = DataLoader(train_manager, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # after_loader = DataLoader(after_manager, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Dataset Prepared: {} train data\n".format(len(train_manager)))

    Task_N = len(train_manager)
    # train init
    paramModel = BaseCostParamModel(feature_len=5,point_len=50).to(device)
    raw_paramModel = copy.deepcopy(paramModel)
    behavior_list = [2,3,4,5,6,8,9]
    paramModels = {}
    optimizers = {}
    for behavior_i in behavior_list:
        paramModels[behavior_i] = copy.deepcopy(paramModel)
        optimizers[behavior_i] = optim.Adam(paramModels[behavior_i].parameters(), lr = args.learning_rate)
    # train_size = len(train_loader.dataset)
    start_time = time.time()
    traj_policy = TrajPolicy().to(device)
    traj_policy.load_state_dict(torch.load(args.model_path, map_location=device))
    traj_policy.eval()

    # prediction, plan, ref_line, current_state, ground_truth
    for task_epoch in range(1, args.task_epochs+1):
        raw_metrics_list = []
        before_metrics_list = []
        after_metrics_list = []
        init_metrics_list = []
        # logging.info(f"\n==========================task: {task_epoch}============================")
        for task_i, x_spt in enumerate(train_loader):
            logging.info(f"===================epoch:{task_epoch}/{args.task_epochs} task_i:{task_i}/{Task_N}======================")
            """train"""
            # prepare data
            ref_line = x_spt[0][0].to(device)
            current_state = x_spt[1][0].to(device)
            ground_truth = x_spt[2][0].to(device)
            behavior_i = x_spt[3][0].item()
            # mean_ground_truth = torch.mean(ground_truth[:, 0,:,:],dim=0).unsqueeze(0)

            # mean_ground_truth = torch.min(ground_truth[:, 0,:,:],dim=0)[0]
            # print("ref_line",ref_line.shape) 
            # print("current_state",current_state.shape)
            # print("ground_truth",ground_truth.shape)
            # print("comfort_cost",comfort_cost.shape)
            # print("efficiency_cost",efficiency_cost.shape)
            """
            ref_line torch.Size([64, 1200, 5])
            current_state torch.Size([64, 11, 8])
            ground_truth torch.Size([64, 11, 50, 5])
            comfort_cost torch.Size([64])
            efficiency_cost torch.Size([64])
            """
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.scheduler_step_size, gamma=0.5)
            E_expert_features = cal_traj_features(ground_truth[:, 0],ref_line,ground_truth[:,1:],current_state).mean(dim=0)
            # get param
            with torch.no_grad():
                control_var = traj_policy.gaussian_sampling(ground_truth[:, 0], std=0.001)
                # control_var = traj_policy(mean_ground_truth.repeat(ground_truth.shape[0], 1, 1))
                control_var_after = copy.deepcopy(control_var)
            ###################################################################################################
            """before train"""
            with torch.no_grad(): # 节省计算图开销
                _, raw_traj, info_str, _ = LfD_Loss(raw_paramModel,t_planner,E_expert_features,control_var,ground_truth,ref_line,current_state,is_opt=False)
                raw_metrics = msirl_metrics(raw_traj, ground_truth)
                raw_metrics_list.append(raw_metrics)
                logging.info(f"\rraw loss--> "+info_str)
                _, before_traj, info_str, _ = LfD_Loss(raw_paramModel,t_planner,E_expert_features,control_var,ground_truth,ref_line,current_state)
                before_metrics = msirl_metrics(before_traj, ground_truth)
                before_metrics_list.append(before_metrics)
                logging.info(f"\rbefore Train --> "+info_str)
                _, init_traj, info_str, _ = LfD_Loss(paramModels[behavior_i],t_planner,E_expert_features,control_var,ground_truth,ref_line,current_state)
                init_metrics = msirl_metrics(init_traj, ground_truth)
                init_metrics_list.append(init_metrics)
                logging.info(f"\rinit Train --> "+info_str)
            ###################################################################################################
            """train process"""
            for t_epoch in range(args.train_epochs):
            #     with torch.no_grad():
            #         if epoch==args.train_epochs-1:
            #             control_var = control_var_after
            #         else:
            #             control_var = traj_policy.gaussian_sampling(ground_truth[:, 0], std=0.001)
                # learn episode
                for step in range(args.step_N):
                    loss, after_traj, step_info_str, control_var = LfD_Loss(paramModels[behavior_i],planner,E_expert_features,control_var,ground_truth,ref_line,current_state)
                    # update IRL
                    # loss backward
                    optimizers[behavior_i].zero_grad() 
                    loss.backward() # retain_graph=True 
                    optimizers[behavior_i].step() 
                    print(f"\rTrain-step: {step+1} / {args.step_N} " + step_info_str, end=" ")
                # scheduler.step() 
                ###################################################################################################
                """after train"""
                # with torch.no_grad(): # 节省计算图开销
                #     loss, after_traj, info_str, _ = LfD_Loss(paramModel,t_planner,E_expert_features,control_var_after,ground_truth,prediction,ref_line,current_state)
                ###################################################################################################
            # print()
                # info_str = step_info_str
                print(f"\rTrain-epoch: {t_epoch+1} / {args.train_epochs} " + step_info_str)
            # print(f"\r", end=" ")
            after_metrics = msirl_metrics(after_traj, ground_truth)
            after_metrics_list.append(after_metrics)
            logging.info(f"after Train --> " + step_info_str) # epoch_N: {args.train_epochs} 

        logging.info("#######################################################################################")
        logging.info("#######################################################################################")
        logging.info("In sum: ")
        logging.info(f"""Task Train-Progress: [{task_epoch:>4d}/{args.task_epochs:>4d}]  Time: {(time.time()-start_time)/task_epoch:>.3f}s/epoch""")
        # show metrics
        raw_epoch_metrics = show_metrics(raw_metrics_list,str="raw")
        # show metrics
        before_epoch_metrics = show_metrics(before_metrics_list,str="before")
        # show metrics
        init_epoch_metrics = show_metrics(init_metrics_list,str="init")
        # show metrics
        after_epoch_metrics = show_metrics(after_metrics_list,str="after")


        logging.info("#######################################################################################")
        logging.info("#######################################################################################")
        # save to training log
        log = {
                'task_epoch':task_epoch,
                'raw-plannerADE1': raw_epoch_metrics[0], 'raw-plannerFDE1': raw_epoch_metrics[3], 
                'raw-plannerADE2': raw_epoch_metrics[1], 'raw-plannerFDE2': raw_epoch_metrics[4],
                'raw-plannerADE3': raw_epoch_metrics[2], 'raw-plannerFDE3': raw_epoch_metrics[5],
                'before-plannerADE1': before_epoch_metrics[0], 'before-plannerFDE1': before_epoch_metrics[3], 
                'before-plannerADE2': before_epoch_metrics[1], 'before-plannerFDE2': before_epoch_metrics[4],
                'before-plannerADE3': before_epoch_metrics[2], 'before-plannerFDE3': before_epoch_metrics[5],
                'init-plannerADE1': init_epoch_metrics[0], 'init-plannerFDE1': init_epoch_metrics[3], 
                'init-plannerADE2': init_epoch_metrics[1], 'init-plannerFDE2': init_epoch_metrics[4],
                'init-plannerADE3': init_epoch_metrics[2], 'init-plannerFDE3': init_epoch_metrics[5],
                'after-plannerADE1': after_epoch_metrics[0], 'after-plannerFDE1': after_epoch_metrics[3], 
                'after-plannerADE2': after_epoch_metrics[1], 'after-plannerFDE2': after_epoch_metrics[4],
                'after-plannerADE3': after_epoch_metrics[2], 'after-plannerFDE3': after_epoch_metrics[5]
            }


        if task_epoch == 1:
            with open(f'./training_log/{log_name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{log_name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        if task_epoch%10==0:
            # save weights
            for behavior_i in behavior_list:
                save_path = log_path + f"epoch{task_epoch}_model/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(paramModels[behavior_i].state_dict(),save_path+f"behavior_{behavior_i}.pth")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='LfD_Training')
    parser.add_argument('--cost_type', type=str, help='type of cost function [lfd/single]', default='lfd')
    parser.add_argument('--train_path', type=str, help='path to train datasets', default='/home/lin/waymo_dataset/task_train')
    # parser.add_argument('--after_path', type=str, help='path to validation datasets', default='/home/lin/waymo_dataset/task_test')
    parser.add_argument('--model_path', type=str, help='path to saved model', default='/home/lin/waymo_dataset/model_param/policy100.pth')
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument('--num_workers', type=int, help="number of workers used for dataloader", default=8)
    # parser.add_argument('--scheduler_step_size', type=int, help='epochs of learning rate ', default=20)
    parser.add_argument('--step_N', type=int, help='epochs of task training', default=3)
    parser.add_argument('--task_epochs', type=int, help='epochs of task training', default=10)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=1) # 20
    # parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=10)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=1e-2)
    parser.add_argument('--gpu_id', type=int, help='run on which device (default: cuda)', default=0)
    args = parser.parse_args()

    # Run
    train_task()
