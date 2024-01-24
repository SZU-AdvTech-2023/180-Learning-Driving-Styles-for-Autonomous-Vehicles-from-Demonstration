
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

# 设置打印选项
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.3f}'.format})

def train_task():
    # device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    # Logging
    log_name = __file__.split('/')[-1].split('.')[0] + f"_{args.seed}"
    log_path = f"./training_log/{log_name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+log_name+'_train.log')

    # set seed
    fixed_seed(args.seed)
    logging.info("------------- {} -------------".format(log_name))
    # logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(device))
    logging.info("scheduler_step_size: {}".format(args.scheduler_step_size))
    logging.info("task_epochs: {}".format(args.task_epochs))
    logging.info("train_epochs: {}".format(args.train_epochs))
    logging.info("test_epochs: {}".format(args.test_epochs))

    # set up data loaders
    train_manager = DataManager_Task(args.train_path, log_path)
    test_manager = DataManager_Task(args.test_path, log_path)
    train_loader = DataLoader(train_manager, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_manager, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # logging.info("Dataset Prepared: {} train data\n".format(len(train_manager)))
    logging.info("Dataset Prepared: {} train data, {} test data\n".format(len(train_manager), len(test_manager)))

    # test_epoch_loss = []
    # test_epoch_metrics_list = []
    # current = 0
    # error_count = 0
    train_size = len(train_loader.dataset)
    start_time = time.time()
    traj_policy = TrajPolicy().to(device)
    optimizer = optim.Adam(traj_policy.parameters(), lr = args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.scheduler_step_size, gamma=0.5)
    # prediction, plan, ref_line, current_state, ground_truth
    for task_epoch in range(1, args.task_epochs+1):
        epoch_loss = []
        epoch_irl_loss = []
        epoch_plan_loss = []
        epoch_metrics_list = []
        logging.info(f"\n==========================task: {task_epoch}============================")
        for x_spt, x_qry in zip(train_loader, test_loader):
            """train"""
            # prepare data
            # prediction = x_spt[0][0].to(device)
            # plan = x_spt[1].to(device)
            ref_line = x_spt[0].to(device)[:,::2,:]
            current_state = x_spt[1].to(device)
            ground_truth = x_spt[2].to(device)

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
            # control_input = torch.ones((7,)).to(device)
            # point_w_input = torch.ones((47,)).to(device)
            # reward_function = RewardFunction().to(device)
            # point_w_param = torch.ones((7, 47)).to(device)
            # point_w_param.requires_grad = True
            # point_w = point_w_param * point_w_input
            
            E_expert_features = cal_traj_features(ground_truth[:, 0],ref_line,ground_truth[:,1:],current_state).mean(dim=0)
            # set up optimizer
            # train process
            for epoch in range(args.train_epochs):
                # get param
                control_var = traj_policy(ground_truth[:, 0])
                # IRL
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
                IRL_loss = F.smooth_l1_loss(E_learner_features, E_expert_features.detach())
                plan_loss = F.smooth_l1_loss(traj, ground_truth[:, 0, :, :2]) 
                plan_loss += F.smooth_l1_loss(traj[:, -1], ground_truth[:, 0, -1, :2])
                loss = IRL_loss + plan_loss
                # update point w
                # cost_point_w = batch_cost_point_w.mean(dim=0)
                # update IRL
                # loss backward
                optimizer.zero_grad() 
                loss.backward() # retain_graph=True
                optimizer.step() 
                # reduce learning rate
                # print(f"\rTrain-epoch: {epoch+1} / {args.train_epochs} loss: {loss.item():.3f} IRL-loss: {IRL_loss.item():.3f}  plan-loss: {plan_loss.item():.3f}  plan-cost: {plan_cost.item():.3f}",end=" ")
                # if error_count>0:
                #     logging.info(f'\n skip-error times:{error_count}')
                #     error_count = 0
                # print("check")
                # except:
                #     error_count +=1
            # logging.info(f"\nTrain-epoch: {epoch+1} / {args.train_epochs} loss: {loss.item():.3f} IRL-loss: {IRL_loss.item():.3f}  plan-loss: {plan_loss.item():.3f}")
            # compute metrics
            metrics = msirl_metrics(traj, ground_truth)
            epoch_metrics_list.append(metrics)
            epoch_irl_loss.append(IRL_loss.item())
            epoch_plan_loss.append(plan_loss.item())
            epoch_loss.append(loss.item())
            # show loss

            """test"""
            # t_prediction = x_qry[0][0].to(device)
            # t_plan = x_qry[1].to(device)
            t_ref_line = x_qry[0].to(device)[:,::2,:]
            t_current_state = x_qry[1].to(device)
            t_ground_truth = x_qry[2].to(device)

            # param init
            # t_optimizer = optim.Adam(traj_policy.parameters(), lr = args.learning_rate)
            # control_variables = torch.zeros((ground_truth.shape[0],7)).to(device)
            E_expert_features = cal_traj_features(t_ground_truth[:,0],t_ref_line,t_ground_truth[:,1:],t_current_state).mean(dim=0)
            for epoch in range(args.test_epochs):
                # get param
                control_var = traj_policy(t_ground_truth[:,0])
                lat_loc = cal_quintic_spline_point(control_var[:,:4], t_current_state[:,0 ,0], t_current_state[:,0 ,3])
                lon_loc = cal_quintic_spline_point(control_var[:,4:], t_current_state[:,0 ,1], t_current_state[:,0 ,4])
                traj = torch.stack([lat_loc, lon_loc],dim=-1)

                E_learner_features = cal_traj_features(traj, t_ref_line, t_ground_truth[:,1:], t_current_state).mean(dim=0)

                # print(E_learner_features.data)
                # print(E_expert_features.data)
                # check
                if torch.isnan(E_learner_features).any().item() or torch.isnan(E_expert_features).any().item():
                    raise Exception("There are Nan in IRL_loss !!!")
                # IRL_loss = F.smooth_l1_loss(E_learner_features, E_expert_features.detach())
                IRL_loss = F.smooth_l1_loss(E_learner_features, E_expert_features.detach())
                plan_loss = F.smooth_l1_loss(traj, t_ground_truth[:, 0, :, :2]) 
                plan_loss += F.smooth_l1_loss(traj[:, -1], t_ground_truth[:, 0, -1, :2])
                loss = IRL_loss + plan_loss
                # loss backward
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                # print(f"\rTest-epoch: {epoch+1} / {args.train_epochs} loss: {loss.item():.3f} IRL-loss: {IRL_loss.item():.3f}  plan-loss: {plan_loss.item():.3f}  plan-cost: {plan_cost.item():.3f}",end=" ")

            # logging.info(f"\nTest-epoch: {epoch+1} / {args.train_epochs} loss: {loss.item():.3f} IRL-loss: {IRL_loss.item():.3f}  plan-loss: {plan_loss.item():.3f}")
            # loss backward
            metrics = msirl_metrics(traj, t_ground_truth)
            epoch_metrics_list.append(metrics)
            epoch_irl_loss.append(IRL_loss.item())
            epoch_plan_loss.append(plan_loss.item())
            epoch_loss.append(loss.item())

        
        logging.info(f"""\nTask Train-Progress: [{task_epoch:>4d}/{args.task_epochs:>4d}]  Time: {(time.time()-start_time)/task_epoch:>.3f}s/task""")
        logging.info(f"\nloss: {np.mean(epoch_loss):.3f} IRL-loss: {np.mean(epoch_irl_loss):.3f}  plan-loss: {np.mean(epoch_plan_loss):.3f}")

        scheduler.step() 
        # show metrics
        epoch_metrics = np.array(epoch_metrics_list)
        plannerADE1, plannerFDE1 = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 3])
        plannerADE2, plannerFDE2 = np.mean(epoch_metrics[:, 1]), np.mean(epoch_metrics[:, 4])
        plannerADE3, plannerFDE3 = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 5])
        epoch_metrics = [plannerADE1, plannerADE2, plannerADE3, plannerFDE1, plannerFDE2, plannerFDE3]
        logging.info(f'\nplannerADE1: {plannerADE1:.4f}, plannerFDE1: {plannerFDE1:.4f}')
        logging.info(f'\nplannerADE2: {plannerADE2:.4f}, plannerFDE2: {plannerFDE2:.4f}')
        logging.info(f'\nplannerADE3: {plannerADE3:.4f}, plannerFDE3: {plannerFDE3:.4f}')
    
        # save to training log
        log = {
                'epoch':task_epoch,
                'loss': np.mean(epoch_loss), 'irl-loss': np.mean(epoch_irl_loss), 'plan-loss': np.mean(epoch_plan_loss), 
                'train-plannerADE1': epoch_metrics[0], 'train-plannerFDE1': epoch_metrics[3], 
                'train-plannerADE2': epoch_metrics[1], 'train-plannerFDE2': epoch_metrics[4],
                'train-plannerADE3': epoch_metrics[2], 'train-plannerFDE3': epoch_metrics[5]
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
                
        if task_epoch%50==0:
            # save weights
            torch.save(traj_policy.state_dict(),log_path+f"policy{task_epoch}.pth")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='LfD_Training')
    parser.add_argument('--train_path', type=str, help='path to train datasets', default='/home/lin/waymo_dataset/traj_train')
    parser.add_argument('--test_path', type=str, help='path to validation datasets', default='/home/lin/waymo_dataset/traj_test')
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument("--num_workers", type=int, help="number of workers used for dataloader", default=8)
    parser.add_argument('--scheduler_step_size', type=int, help='epochs of learning rate ', default=20)
    parser.add_argument('--task_epochs', type=int, help='epochs of task training', default=100)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=1)
    parser.add_argument('--test_epochs', type=int, help='epochs of testing', default=1)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=256)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=1e-3)
    parser.add_argument('--gpu_id', type=int, help='run on which device (default: cuda)', default=0)
    args = parser.parse_args()

    # Run
    train_task()
