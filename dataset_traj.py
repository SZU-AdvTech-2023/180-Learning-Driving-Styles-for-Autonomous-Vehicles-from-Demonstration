import torch
import argparse
import os
import logging
import time
from utils.test_utils import *
from model.Pipeline import Predictor
from model.DataManager import *
from torch.utils.data import DataLoader
import sys
# from utils.train_utils import *

def select_future_traj(plans, predictions, scores):
    best_mode = torch.argmax(scores, dim=-1)
    plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

    return plan, prediction

def convert2np(*data:torch.Tensor):
    res = []
    for elem in data:
        res.append(elem.cpu().numpy())
    return tuple(res)

def data_generation():
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    # logging
    # log_path = f"./traj_log/{args.name}/"
    # os.makedirs(log_path, exist_ok=True)
    # initLogging(log_file=log_path+'traj_processing.log')

    # logging.info("------------- {} -------------".format(args.name))
    # logging.info("Use gpu_id: {}".format(args.gpu_id))
    # logging.info("model_path: {}".format(args.model_path))

    # process file
    manager = DataManager_Traj(args.process_path + 'raw_' + args.name,args.save_path + 'traj_' + args.name)
    data_loader = DataLoader(manager, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load model
    predictor = Predictor(50).to(device)
    predictor.load_state_dict(torch.load(args.model_path, map_location=device))
    predictor.eval()
    
    current = 0
    size = len(data_loader.dataset)
    start_time = time.time()

    for batch in data_loader:
        # prepare data      
        ego = batch[0].to(device)
        neighbors = batch[1].to(device)
        lanes = batch[2].to(device)
        crosswalks = batch[3].to(device)
        ref_line = batch[4].to(device)
        ground_truth = batch[5].to(device)
        scene_id = list(batch[6])
        time_step = batch[7].to(device)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

        # predict
        with torch.no_grad():
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, lanes, crosswalks)
            plan, prediction = select_future_traj(plans, predictions, scores)
        # plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

        # plan = plan.cpu().numpy()[0]
        # logging.info(f"Prediction ADE: {prediction_error[0]}, FDE: {prediction_error[1]}")
        
        # data save
        time_step, ego, prediction, plan, ref_line, current_state, ground_truth = convert2np(time_step, ego, prediction, plan, ref_line, current_state, ground_truth)
        
        # ego = ego.cpu().numpy()

        # print()
        # print("prediction: ",prediction.shape)
        # print("plan: ",plan.shape)
        # print("ref_line: ",ref_line.shape)
        # print("current_state: ",current_state.shape)
        # print("ground_truth: ",ground_truth.shape)
        # print("time_step: ",time_step.shape)
        """
        prediction:  (128, 10, 50, 3)
        plan:  (128, 50, 2)
        ref_line:  (128, 1200, 5)
        current_state:  (128, 11, 8)
        ground_truth:  (128, 11, 50, 5)
        time_step:  (128,)
        comfort_cost:  (128,)
        efficiency_cost:  (128,)
        action_type:  (128,)
        """

        # show progress
        current += batch[0].shape[0]
        sys.stdout.write(f"\rprogress:[{current:>6d}/{size:>6d}] time:{(time.time()-start_time)/current:>.3f}s/sample")
        sys.stdout.flush()
        # save_data
        manager.save_data(scene_id, time_step, ego, prediction, plan, ref_line, current_state, ground_truth)
    # save results
    # df.to_csv(f'./processing_log/{args.name}/processing_log.csv')

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--name', type=str, help='log name (default: "Process")', default="test")
    parser.add_argument('--process_path', type=str, help='path to processing datasets', default='/home/lin/waymo_dataset/')
    parser.add_argument("--num_workers", type=int, help="number of workers used for dataloader", default=8)
    parser.add_argument('--model_path', type=str, help='path to saved model', default='/home/lin/waymo_dataset/model_param/model_5_0.7052.pth')
    parser.add_argument('--save_path', type=str, help='path to mid result', default='/home/lin/waymo_dataset/')
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--gpu_id', type=str, help='run on which gpu (default: cpu)', default='0')
    args = parser.parse_args()

    # Run
    data_generation()
