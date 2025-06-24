import os
import shutil
import argparse
import emage.mertic
from moviepy.tools import verbose_print
from omegaconf import OmegaConf
import random
import numpy as np
import json 
import librosa
from datetime import datetime

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
import wandb
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import smplx
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import igraph

import emage
import utils.rotation_conversions as rc
from create_graph import path_visualization, graph_pruning, get_motion_reps_tensor

def search_path(graph, audio_low_np, audio_high_np, top_k=1, loop_penalty=0.1, search_mode="both"):
    T = audio_low_np.shape[0]  # Total time steps
    # Initialize the beam with start nodes (nodes with no previous node)
    start_nodes = [v for v in graph.vs if v['previous'] is None or v['previous'] == -1]
    beam = []
    for node in start_nodes:
        motion_low = node['motion_low']  # Shape: [C]
        motion_high = node['motion_high']  # Shape: [C]
        # cost = np.linalg.norm(audio_low_np[0] - motion_low) + np.linalg.norm(audio_high_np - motion_high)
        if search_mode == "both":
            cost = 2 - (np.dot(audio_low_np[0], motion_low.T) + np.dot(audio_high_np[0], motion_high.T))
        elif search_mode == "high_level":
            cost = 1 - np.dot(audio_high_np[0], motion_high.T)
        elif search_mode == "low_level":
            cost = 1 - np.dot(audio_low_np[0], motion_low.T)
        sequence = [node]
        beam.append((cost, sequence))

    # Keep only the top_k initial nodes
    beam.sort(key=lambda x: x[0])
    beam = beam[:top_k]

    # Beam search over time steps
    for t in range(1, T):
        new_beam = []
        for cost, seq in beam:
            last_node = seq[-1]
            neighbor_indices = graph.neighbors(last_node.index, mode='OUT')
            if not neighbor_indices:
                continue  # No outgoing edges from the last node
            for idx in neighbor_indices:
                neighbor = graph.vs[idx]
                # Check for loops
                if neighbor in seq:
                    # Apply loop penalty
                    loop_cost = cost + loop_penalty
                else:
                    loop_cost = cost

                motion_low = neighbor['motion_low']  # Shape: [C]
                motion_high = neighbor['motion_high']  # Shape: [C]
                # cost_increment = np.linalg.norm(audio_low_np[t] - motion_low) + np.linalg.norm(audio_high_np[t] - motion_high)
                if search_mode == "both":
                    cost_increment = 2 - (np.dot(audio_low_np[t], motion_low.T) + np.dot(audio_high_np[t], motion_high.T))
                elif search_mode == "high_level":
                    cost_increment = 1 - np.dot(audio_high_np[t], motion_high.T)
                elif search_mode == "low_level":
                    cost_increment = 1 - np.dot(audio_low_np[t], motion_low.T)
                new_cost = loop_cost + cost_increment
                new_seq = seq + [neighbor]
                new_beam.append((new_cost, new_seq))
        if not new_beam:
            break  # Cannot extend any further
        # Keep only the top_k sequences
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:top_k]

    # Extract paths and continuity information
    path_list = []
    is_continue_list = []
    for cost, seq in beam:
        path_list.append(seq)
        print("Cost: ", cost, "path", [node.index for node in seq])
        is_continue = []
        for i in range(len(seq) - 1):
            edge_id = graph.get_eid(seq[i].index, seq[i + 1].index)
            is_cont = graph.es[edge_id]['is_continue']
            is_continue.append(is_cont)
        is_continue_list.append(is_continue)
    return path_list, is_continue_list

def search_path_dp(graph, audio_low_np, audio_high_np, loop_penalty=0.01, top_k=1, search_mode="both", continue_penalty=0.01):
    T = audio_low_np.shape[0]  # Total time steps
    N = len(graph.vs)          # Total number of nodes in the graph

    # Initialize DP tables
    min_cost = [{} for _ in range(T)]         # min_cost[t][node.index] = (cost, predecessor_index, non_continue_count)
    visited_nodes = [{} for _ in range(T)]    # visited_nodes[t][node.index] = dict of node visit counts

    # Initialize the first time step
    start_nodes = [v for v in graph.vs if v['previous'] is None or v['previous'] == -1]
    for node in start_nodes:
        motion_low = node['motion_low']      # Shape: [C]
        motion_high = node['motion_high']    # Shape: [C]

        # Cost using cosine similarity
        if search_mode == "both":
            cost = 2 - (np.dot(audio_low_np[0], motion_low.T) + np.dot(audio_high_np[0], motion_high.T))
        elif search_mode == "high_level":
            cost = 1 - np.dot(audio_high_np[0], motion_high.T)
        elif search_mode == "low_level":
            cost = 1 - np.dot(audio_low_np[0], motion_low.T)
        
        min_cost[0][node.index] = (cost, None, 0)  # Initialize with no predecessor and 0 non-continue count
        visited_nodes[0][node.index] = {node.index: 1}  # Initialize visit count as a dictionary

    # DP over time steps
    for t in range(1, T):
        for node in graph.vs:
            node_index = node.index
            min_cost_t = float('inf')
            best_predecessor = None
            best_visited = None
            best_non_continue_count = 0

            # Incoming edges to the current node
            incoming_edges = graph.es.select(_to=node_index)
            for edge in incoming_edges:
                prev_node_index = edge.source
                prev_node = graph.vs[prev_node_index]
                if prev_node_index in min_cost[t-1]:
                    prev_cost, _, prev_non_continue_count = min_cost[t-1][prev_node_index]
                    prev_visited = visited_nodes[t-1][prev_node_index]

                    # Loop punishment
                    if node_index in prev_visited:
                        loop_time = prev_visited[node_index]  # Get the count of previous visits
                        loop_cost = prev_cost + loop_penalty * np.exp(loop_time)  # Apply exponential penalty
                        new_visited = prev_visited.copy()
                        new_visited[node_index] = loop_time + 1  # Increment visit count
                    else:
                        loop_cost = prev_cost
                        new_visited = prev_visited.copy()
                        new_visited[node_index] = 1  # Initialize visit count for the new node

                    motion_low = node['motion_low']      # Shape: [C]
                    motion_high = node['motion_high']    # Shape: [C]
                    
                    if search_mode == "both":
                        cost_increment = 2 - (np.dot(audio_low_np[t], motion_low.T) + np.dot(audio_high_np[t], motion_high.T))
                    elif search_mode == "high_level":
                        cost_increment = 1 - np.dot(audio_high_np[t], motion_high.T)
                    elif search_mode == "low_level":
                        cost_increment = 1 - np.dot(audio_low_np[t], motion_low.T)
                    
                    # Check if the edge is "is_continue"
                    edge_id = edge.index
                    is_continue = graph.es[edge_id]['is_continue']
                    
                    if not is_continue:
                        non_continue_count = prev_non_continue_count + 1  # Increment the count of non-continue edges
                    else:
                        non_continue_count = prev_non_continue_count

                    # Apply the penalty based on the square of the number of non-continuous edges
                    continue_penalty_cost = continue_penalty * non_continue_count
                    
                    total_cost = loop_cost + cost_increment + continue_penalty_cost

                    if total_cost < min_cost_t:
                        min_cost_t = total_cost
                        best_predecessor = prev_node_index
                        best_visited = new_visited
                        best_non_continue_count = non_continue_count

            if best_predecessor is not None:
                min_cost[t][node_index] = (min_cost_t, best_predecessor, best_non_continue_count)
                visited_nodes[t][node_index] = best_visited  # Store the new visit count dictionary

    # Find the node with the minimal cost at the last time step
    final_min_cost = float('inf')
    final_node_index = None
    for node_index, (cost, _, _) in min_cost[T-1].items():
        if cost < final_min_cost:
            final_min_cost = cost
            final_node_index = node_index

    if final_node_index is None:
        print("No valid path found.")
        return [], []

    # Backtrack to reconstruct the optimal path
    optimal_path_indices = []
    current_node_index = final_node_index
    for t in range(T-1, -1, -1):
        optimal_path_indices.append(current_node_index)
        _, predecessor, _ = min_cost[t][current_node_index]
        current_node_index = predecessor if predecessor is not None else None

    optimal_path_indices = optimal_path_indices[::-1]  # Reverse to get correct order
    optimal_path = [graph.vs[idx] for idx in optimal_path_indices]

    # Extract continuity information
    is_continue = []
    for i in range(len(optimal_path) - 1):
        edge_id = graph.get_eid(optimal_path[i].index, optimal_path[i + 1].index)
        is_cont = graph.es[edge_id]['is_continue']
        is_continue.append(is_cont)

    print("Optimal Cost: ", final_min_cost, "Path: ", optimal_path_indices)
    return [optimal_path], [is_continue]


# from torch.cuda.amp import autocast, GradScaler
# from torch.nn.utils import clip_grad_norm_

# # Initialize GradScaler
# scaler = GradScaler()

def train_val_fn(batch, model, device, mode="train", optimizer=None, lr_scheduler=None, max_grad_norm=1.0, **kwargs):
    if mode == "train":
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
    else:
        model.eval()
        torch.set_grad_enabled(False)

    cached_rep15d = batch["cached_rep15d"].to(device)
    cached_audio_low = batch["cached_audio_low"].to(device)
    cached_audio_high = batch["cached_audio_high"].to(device)
    bert_time_aligned = batch["bert_time_aligned"].to(device)
    cached_audio_high = torch.cat([cached_audio_high, bert_time_aligned], dim=-1)
    audio_tensor = batch["audio_tensor"].to(device)

    # with autocast():  # Mixed precision context
    model_out = model(cached_rep15d=cached_rep15d, cached_audio_low=cached_audio_low, cached_audio_high=cached_audio_high, in_audio=audio_tensor)
    audio_lower = model_out["audio_low"]
    motion_lower = model_out["motion_low"]
    audio_hihger_cls = model_out["audio_cls"]
    motion_higher_cls = model_out["motion_cls"]

    high_loss = model_out["high_level_loss"]
    low_infonce, low_acc = model_out["low_level_loss"]
    loss_dict = {
        "low_cosine": low_infonce,
        "high_infonce": high_loss
    }
    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss
    loss_dict["low_acc"] = low_acc
    loss_dict["acc"] = compute_average_precision(audio_hihger_cls, motion_higher_cls)

    if mode == "train":
        # Use GradScaler for backward pass
        # scaler.scale(loss).backward()
        
        # Clip gradients to the maximum norm
        # scaler.unscale_(optimizer)  # Unscale gradients before clipping
        # clip_grad_norm_(model.parameters(), max_grad_norm)

        # Step the optimizer
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    return loss_dict


def test_fn(model, device, smplx_model, iteration, fgd_fn, srgr_fn, bc_fn, l1div_fn, candidate_json_path, test_path, cfg, **kwargs):
    torch.set_grad_enabled(False)
    pool_path = "./datasets/oliver_test/show-oliver-test.pkl"
    graph = igraph.Graph.Read_Pickle(fname=pool_path)

    save_dir = os.path.join(test_path, f"retrieved_motions_{iteration}")
    os.makedirs(save_dir, exist_ok=True)

    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    actual_model.eval()

    with open(candidate_json_path, 'r') as f:
        candidate_data = json.load(f)
    all_motions = {}
    for i, node in enumerate(graph.vs):
        if all_motions.get(node["name"]) is None:
            all_motions[node["name"]] = [node["axis_angle"].reshape(-1)]
        else:
            all_motions[node["name"]].append(node["axis_angle"].reshape(-1))
    for k, v in all_motions.items():
        all_motions[k] = np.stack(v) # T, J*3
    
    window_size = cfg.data.pose_length
    motion_high_all = []
    motion_low_all = []
    for k, v in all_motions.items():
        motion_tensor = torch.from_numpy(v).float().to(device).unsqueeze(0)
        _, t, _ = motion_tensor.shape

        num_chunks = t // window_size
        motion_high_list = []
        motion_low_list = []

        for i in range(num_chunks):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            motion_slice = motion_tensor[:, start_idx:end_idx, :]
            
            motion_features = actual_model.get_motion_features(motion_slice)
            motion_high = motion_features["motion_high_weight"].cpu().numpy()
            motion_low = motion_features["motion_low"].cpu().numpy()

            motion_high_list.append(motion_high[0])
            motion_low_list.append(motion_low[0])

        remain_length = t % window_size
        if remain_length > 0:
            start_idx = t - window_size
            motion_slice = motion_tensor[:, start_idx:, :]

            motion_features = actual_model.get_motion_features(motion_slice)
            motion_high = motion_features["motion_high_weight"].cpu().numpy()
            motion_low = motion_features["motion_low"].cpu().numpy()

            motion_high_list.append(motion_high[0][-remain_length:])
            motion_low_list.append(motion_low[0][-remain_length:])

        motion_high_all.append(np.concatenate(motion_high_list, axis=0))
        motion_low_all.append(np.concatenate(motion_low_list, axis=0))

    motion_high_all = np.concatenate(motion_high_all, axis=0)
    motion_low_all = np.concatenate(motion_low_all, axis=0)
    # print(motion_high_all.shape, motion_low_all.shape)
    motion_low_all = motion_low_all / np.linalg.norm(motion_low_all, axis=1, keepdims=True)
    motion_high_all = motion_high_all / np.linalg.norm(motion_high_all, axis=1, keepdims=True)
    assert motion_high_all.shape[0] == len(graph.vs)
    assert motion_low_all.shape[0] == len(graph.vs)
    
    for i, node in enumerate(graph.vs):
        node["motion_high"] = motion_high_all[i]
        node["motion_low"] = motion_low_all[i]
    graph = graph_pruning(graph)

    for idx, pair in enumerate(tqdm(candidate_data, desc="Testing")):
        gt_motion = np.load(pair["motion_path"] + ".npz", allow_pickle=True)["poses"]
        target_length = gt_motion.shape[0]
        audio_path = pair["audio_path"] + ".wav"
        audio_waveform, sr = librosa.load(audio_path)
        audio_waveform = librosa.resample(audio_waveform, orig_sr=sr, target_sr=cfg.data.audio_sr)
        audio_tensor = torch.from_numpy(audio_waveform).float().to(device).unsqueeze(0)

        window_size = int(cfg.data.audio_sr * (cfg.data.pose_length / 30))
        _, t = audio_tensor.shape

        num_chunks = t // window_size
        audio_low_list = []
        audio_high_list = []

        for i in range(num_chunks):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            # print(start_idx, end_idx, window_size)
            audio_slice = audio_tensor[:, start_idx:end_idx]

            model_out_candidates = actual_model.get_audio_features(audio_slice)
            audio_low = model_out_candidates["audio_low"]
            audio_high = model_out_candidates["audio_high_weight"]

            audio_low = F.normalize(audio_low, dim=2)[0].cpu().numpy()
            audio_high = F.normalize(audio_high, dim=2)[0].cpu().numpy()

            audio_low_list.append(audio_low)
            audio_high_list.append(audio_high)
            # print(audio_low.shape, audio_high.shape)
        

        remain_length = t % window_size
        if remain_length > 0:
            start_idx = t - window_size
            audio_slice = audio_tensor[:, start_idx:]

            model_out_candidates = actual_model.get_audio_features(audio_slice)
            audio_low = model_out_candidates["audio_low"]
            audio_high = model_out_candidates["audio_high_weight"]
            
            gap = target_length - np.concatenate(audio_low_list, axis=0).shape[1]
            audio_low = F.normalize(audio_low, dim=2)[0][-gap:].cpu().numpy()
            audio_high = F.normalize(audio_high, dim=2)[0][-gap:].cpu().numpy()
            
            # print(audio_low.shape, audio_high.shape)
            audio_low_list.append(audio_low)
            audio_high_list.append(audio_high)

        audio_low_all = np.concatenate(audio_low_list, axis=0)
        audio_high_all = np.concatenate(audio_high_list, axis=0)
        # search the path with audio low features [T, c] and audio high features [T, c]
        path_list, is_continue_list = search_path(graph, audio_low_all, audio_high_all, top_k=1, search_mode="high_level")
        res_motion = []
        counter = 0
        for path, is_continue in zip(path_list, is_continue_list):
            res_motion_current = path_visualization(
              graph, path, is_continue, os.path.join(save_dir, f"audio_{idx}_retri_{counter}.mp4"), audio_path=audio_path, return_motion=True, verbose_continue=True
            )
            res_motion.append(res_motion_current)
            np.savez(os.path.join(save_dir, f"audio_{idx}_retri_{counter}.npz"), motion=res_motion_current)
            counter += 1

    metrics = {}
    counts = {"top1": 0, "top3": 0, "top10": 0}
    
    fgd_fn.reset()
    l1div_fn.reset()
    bc_fn.reset()
    srgr_fn.reset()
    for idx, pair in enumerate(tqdm(candidate_data, desc="Evaluating")):
        gt_motion = np.load(pair["motion_path"] + ".npz", allow_pickle=True)["poses"]
        audio_path = pair["audio_path"] + ".wav"
        gt_motion_tensor = torch.from_numpy(gt_motion).float().to(device).unsqueeze(0)
        bs, n, _ = gt_motion_tensor.size()
        audio_waveform, sr = librosa.load(audio_path, sr=None)
        audio_waveform = librosa.resample(audio_waveform, orig_sr=sr, target_sr=cfg.data.audio_sr)
        audio_tensor = torch.from_numpy(audio_waveform).float().to(device).unsqueeze(0)

        top1_path = os.path.join(save_dir, f"audio_{idx}_retri_0.npz")
        top1_motion = np.load(top1_path, allow_pickle=True)["motion"] # T 165
        top1_motion_tensor = torch.from_numpy(top1_motion).float().to(device).unsqueeze(0) # Add bs, to 1 T 165
            
        gt_vertex = smplx_model(
            betas=torch.zeros(bs*n, 300).to(device),
            transl=torch.zeros(bs*n, 3).to(device),
            expression=torch.zeros(bs*n, 100).to(device),
            jaw_pose=torch.zeros(bs*n, 3).to(device),
            global_orient=torch.zeros(bs*n, 3).to(device),
            body_pose=gt_motion_tensor.reshape(bs*n, 55*3)[:, 3:21*3+3],
            left_hand_pose=gt_motion_tensor.reshape(bs*n, 55*3)[:, 25*3:40*3],
            right_hand_pose=gt_motion_tensor.reshape(bs*n, 55*3)[:, 40*3:55*3],
            return_joints=True,
            leye_pose=torch.zeros(bs*n, 3).to(device),
            reye_pose=torch.zeros(bs*n, 3).to(device),
        )["joints"].detach().cpu().numpy().reshape(bs, n, 127*3)[0, :, :55*3]
        top1_vertex = smplx_model(
            betas=torch.zeros(bs*n, 300).to(device),
            transl=torch.zeros(bs*n, 3).to(device),
            expression=torch.zeros(bs*n, 100).to(device),
            jaw_pose=torch.zeros(bs*n, 3).to(device),
            global_orient=torch.zeros(bs*n, 3).to(device),
            body_pose=top1_motion_tensor.reshape(bs*n, 55*3)[:, 3:21*3+3],
            left_hand_pose=top1_motion_tensor.reshape(bs*n, 55*3)[:, 25*3:40*3],
            right_hand_pose=top1_motion_tensor.reshape(bs*n, 55*3)[:, 40*3:55*3],
            return_joints=True,
            leye_pose=torch.zeros(bs*n, 3).to(device),
            reye_pose=torch.zeros(bs*n, 3).to(device),
        )["joints"].detach().cpu().numpy().reshape(bs, n, 127*3)[0, :, :55*3]
    
        l1div_fn.run(top1_vertex)
        # print(audio_waveform.shape, top1_vertex.shape)
        onset_bt = bc_fn.load_audio(audio_waveform, t_start=None, without_file=True, sr_audio=cfg.data.audio_sr)
        beat_vel = bc_fn.load_pose(top1_vertex, 0, n, pose_fps = 30, without_file=True)
        # print(n)
        # print(onset_bt)
        # print(beat_vel)
        bc_fn.calculate_align(onset_bt, beat_vel, 30)
        srgr_fn.run(gt_vertex, top1_vertex)
        
        gt_motion_tensor = rc.axis_angle_to_matrix(gt_motion_tensor.reshape(1, n, 55, 3))
        gt_motion_tensor = rc.matrix_to_rotation_6d(gt_motion_tensor).reshape(1, n, 55*6)
        top1_motion_tensor = rc.axis_angle_to_matrix(top1_motion_tensor.reshape(1, n, 55, 3))
        top1_motion_tensor = rc.matrix_to_rotation_6d(top1_motion_tensor).reshape(1, n, 55*6)
        remain = n % 32
        if remain != 0:
          gt_motion_tensor = gt_motion_tensor[:, :n-remain]
          top1_motion_tensor = top1_motion_tensor[:, :n-remain]
        # print(gt_motion_tensor.shape, top1_motion_tensor.shape)
        fgd_fn.update(gt_motion_tensor, top1_motion_tensor)
       
    metrics["fgd_top1"] = fgd_fn.compute()
    metrics["l1_top1"] = l1div_fn.avg()
    metrics["bc_top1"] = bc_fn.avg()
    metrics["srgr_top1"] = srgr_fn.avg()

    print(f"Test Metrics at Iteration {iteration}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    return metrics


def compute_average_precision(feature1, feature2):
    # Normalize the features
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)
    
    # Compute the similarity matrix
    similarity_matrix = torch.matmul(feature1, feature2.t())
    
    # Get the top-1 predicted indices for each feature in feature1
    top1_indices = torch.argmax(similarity_matrix, dim=1)
    
    # Generate ground truth labels (diagonal indices)
    batch_size = feature1.size(0)
    ground_truth = torch.arange(batch_size, device=feature1.device)
    
    # Compute the accuracy (True if the top-1 index matches the ground truth)
    correct_predictions = (top1_indices == ground_truth).float()
    
    # Compute average precision
    average_precision = correct_predictions.mean()
    
    return average_precision


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, output1, output2):
        # Calculate cosine similarity
        cosine_sim = self.cosine_similarity(output1, output2)
        # Loss is 1 minus the average cosine similarity
        return 1 - cosine_sim.mean()

class InfoNCELossCross(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELossCross, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feature1, feature2):
        """
        Args:
            feature1: tensor of shape (batch_size, feature_dim)
            feature2: tensor of shape (batch_size, feature_dim)
                      where each corresponding index in feature1 and feature2 is a positive pair,
                      and all other combinations are negative pairs.
        """
        batch_size = feature1.size(0)
        
        # Normalize feature vectors
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        
        # Compute similarity matrix between feature1 and feature2
        similarity_matrix = torch.matmul(feature1, feature2.t()) / self.temperature
        
        # Labels for each element in feature1 are the indices of their matching pairs in feature2
        labels = torch.arange(batch_size, device=feature1.device)
        
        # Cross entropy loss for each positive pair with all corresponding negatives
        loss = self.criterion(similarity_matrix, labels)
        return loss


class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, motion_feature, audio_feature, learned_temp=None):
        if learned_temp is not None:
            temperature = learned_temp
        else:
            temperature = self.temperature
        batch_size, T, _ = motion_feature.size()
        assert len(motion_feature.shape) == 3

        motion_feature = F.normalize(motion_feature, dim=2)
        audio_feature = F.normalize(audio_feature, dim=2)

        motion_to_audio_loss = 0
        audio_to_motion_loss = 0
        motion_to_audio_correct = 0
        audio_to_motion_correct = 0

        # First pass: motion to audio
        for t in range(T):
            motion_feature_t = motion_feature[:, t, :]  # (bs, c)

            # Positive pair range for motion
            start = max(0, t - 4)
            end = min(T, t + 4)
            positive_audio_feature = audio_feature[:, start:end, :]  # (bs, pos_range, c)

            # Negative pair range for motion
            left_end = start
            left_start = max(0, left_end - 4 * 3)
            right_start = end
            right_end = min(T, right_start + 4 * 3)
            negative_audio_feature = torch.cat(
                [audio_feature[:, left_start:left_end, :], audio_feature[:, right_start:right_end, :]],
                dim=1
            )  # (bs, neg_range, c)

            # Concatenate positive and negative samples
            combined_audio_feature = torch.cat([positive_audio_feature, negative_audio_feature], dim=1)  # (bs, pos_range + neg_range, c)

            # Compute similarity scores
            logits = torch.matmul(motion_feature_t.unsqueeze(1), combined_audio_feature.transpose(1, 2)) / temperature  # (bs, 1, pos_range + neg_range)
            logits = logits.squeeze(1)  # (bs, pos_range + neg_range)

            # Compute InfoNCE loss
            positive_scores = logits[:, :positive_audio_feature.size(1)]
            loss_t = -positive_scores.logsumexp(dim=1) + torch.logsumexp(logits, dim=1)
            motion_to_audio_loss += loss_t.mean()

            # Compute accuracy
            max_indices = torch.argmax(logits, dim=1)
            correct_mask = (max_indices < positive_audio_feature.size(1)).float()  # Check if indices are within the range of positive samples
            motion_to_audio_correct += correct_mask.sum()

        # Second pass: audio to motion
        for t in range(T):
            audio_feature_t = audio_feature[:, t, :]  # (bs, c)

            # Positive pair range for audio
            start = max(0, t - 4)
            end = min(T, t + 4)
            positive_motion_feature = motion_feature[:, start:end, :]  # (bs, pos_range, c)

            # Negative pair range for audio
            left_end = start
            left_start = max(0, left_end - 4 * 3)
            right_start = end
            right_end = min(T, right_start + 4 * 3)
            negative_motion_feature = torch.cat(
                [motion_feature[:, left_start:left_end, :], motion_feature[:, right_start:right_end, :]],
                dim=1
            )  # (bs, neg_range, c)

            # Concatenate positive and negative samples
            combined_motion_feature = torch.cat([positive_motion_feature, negative_motion_feature], dim=1)  # (bs, pos_range + neg_range, c)

            # Compute similarity scores
            logits = torch.matmul(audio_feature_t.unsqueeze(1), combined_motion_feature.transpose(1, 2)) / temperature  # (bs, 1, pos_range + neg_range)
            logits = logits.squeeze(1)  # (bs, pos_range + neg_range)

            # Compute InfoNCE loss
            positive_scores = logits[:, :positive_motion_feature.size(1)]
            loss_t = -positive_scores.logsumexp(dim=1) + torch.logsumexp(logits, dim=1)
            audio_to_motion_loss += loss_t.mean()

            # Compute accuracy
            max_indices = torch.argmax(logits, dim=1)
            correct_mask = (max_indices < positive_motion_feature.size(1)).float()  # Check if indices are within the range of positive samples
            audio_to_motion_correct += correct_mask.sum()


        # Average the two losses
        final_loss = (motion_to_audio_loss + audio_to_motion_loss) / (2 * T)

        # Compute final accuracy
        total_correct = (motion_to_audio_correct + audio_to_motion_correct) / (2 * T * batch_size)
        
        return final_loss, total_correct



class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature1, feature2, learned_temp=None):
        batch_size = feature1.size(0)
        assert len(feature1.shape) == 2
        if learned_temp is not None:
            temperature = learned_temp
        else:
            temperature = self.temperature
        # Normalize feature vectors
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        # Compute similarity matrix between feature1 and feature2
        similarity_matrix = torch.matmul(feature1, feature2.t()) / temperature
        # Extract positive similarities (diagonal elements)
        positive_similarities = torch.diag(similarity_matrix)
        # Compute the denominator using logsumexp for numerical stability
        denominator = torch.logsumexp(similarity_matrix, dim=1)
        # Compute the InfoNCE loss
        loss = - (positive_similarities - denominator).mean()
        return loss
      

def main(cfg):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0  

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl")
    seed_everything(cfg.seed)

    experiment_ckpt_dir = experiment_log_dir = os.path.join(cfg.output_dir, cfg.exp_name)

    smplx_model = smplx.create(
            "./emage/smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(device).eval()

    model = init_class(cfg.model.name_pyfile, cfg.model.class_name, cfg).cuda()
    for param in model.parameters():
        param.requires_grad = True
    # freeze wav2vec2
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    model.smplx_model = smplx_model
    model.get_motion_reps = get_motion_reps_tensor
    model.high_level_loss_fn = InfoNCELoss()
    model.low_level_loss_fn = LocalContrastiveLoss()
      
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
        print("using 8 bit")
    else:
        optimizer_cls = torch.optim.AdamW


    optimizer = optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,)
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    loss_cosine = CosineSimilarityLoss().to(device)
    loss_mse = nn.MSELoss().to(device)
    loss_l1 = nn.L1Loss().to(device)
    loss_infonce = InfoNCELossCross().to(device)
    loss_fn_dict = {
        "loss_cosine": loss_cosine,
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "loss_infonce": loss_infonce,
    }
    
    fgd_fn = emage.mertic.FGD(download_path="./emage/")
    srgr_fn = emage.mertic.SRGR(threshold=0.3, joints=55, joint_dim=3)
    bc_fn = emage.mertic.BC(download_path="./emage/", sigma=0.5, order=7)
    l1div_fn = emage.mertic.L1div()

    train_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='train')
    test_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.train_bs, sampler=train_sampler, drop_last=True, num_workers=4)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=256, sampler=test_sampler, drop_last=False, num_workers=4)

    if local_rank == 0:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.exp_name + "_" + run_time,
            entity=cfg.wandb_entity,
            dir=cfg.wandb_log_dir,
            config=OmegaConf.to_container(cfg)  # Pass config directly during initialization
        )
    else:
        writer = None
    
    num_epochs = cfg.solver.max_train_steps // len(train_loader) + 1
    iteration = 0
    val_best = {}
    test_best = {}

    # checkpoint_path = "/content/drive/MyDrive/005_Weights/baseline_high_env0/checkpoint_3800/ckpt.pth"
    # checkpoint = torch.load(checkpoint_path)
    # state_dict = checkpoint['model_state_dict']
    # #new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict, strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    # iteration = checkpoint["iteration"]

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            loss_dict = train_val_fn(
                batch, model, device, mode="train", optimizer=optimizer, lr_scheduler=lr_scheduler,
                loss_fn_dict=loss_fn_dict
            )
            if local_rank == 0 and iteration % cfg.log_period == 0:
                for key, value in loss_dict.items():
                    # writer.add_scalar(f"train/{key}", value, iteration)
                    wandb.log({f"train/{key}": value}, step=iteration)
                loss_message = ", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] - {loss_message}")

            if local_rank == 0 and iteration % cfg.validation.val_loss_steps == 0:
                val_loss_dict = {}
                val_batches = 0
                for batch in tqdm(test_loader):
                    loss_dict = train_val_fn(
                        batch, model, device, mode="val", optimizer=optimizer, lr_scheduler=lr_scheduler,
                        loss_fn_dict=loss_fn_dict
                    )
                    for k, v in loss_dict.items():
                        if k not in val_loss_dict:
                            val_loss_dict[k] = 0
                        val_loss_dict[k] += v.item()  # Convert to float for accumulation
                    val_batches += 1
                    if val_batches == 10:
                        break
                val_loss_mean_dict = {k: v / val_batches for k, v in val_loss_dict.items()}
                for k, v in val_loss_mean_dict.items():
                    if k not in val_best or v > val_best[k]["value"]:
                        val_best[k] = {"value": v, "iteration": iteration}
                        if "acc" in k:
                            checkpoint_path = os.path.join(experiment_ckpt_dir, f"ckpt_{k}")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            torch.save({
                                'iteration': iteration,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            }, os.path.join(checkpoint_path, "ckpt.pth"))

                    print(f"Val [{iteration}] - {k}: {v:.6f} (best: {val_best[k]['value']:.6f} at {val_best[k]['iteration']})")
                    # writer.add_scalar(f"val/{k}", v, iteration)
                    wandb.log({f"val/{k}": v}, step=iteration)
        
                checkpoint_path = os.path.join(experiment_ckpt_dir, f"checkpoint_{iteration}")
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, os.path.join(checkpoint_path, "ckpt.pth"))
                checkpoints = [d for d in os.listdir(experiment_ckpt_dir) if os.path.isdir(os.path.join(experiment_ckpt_dir, d)) and d.startswith("checkpoint_")]
                checkpoints.sort(key=lambda x: int(x.split("_")[1]))
                if len(checkpoints) > 3:
                    for ckpt_to_delete in checkpoints[:-3]:
                        shutil.rmtree(os.path.join(experiment_ckpt_dir, ckpt_to_delete))

            # if local_rank == 0 and iteration % cfg.validation.validation_steps == 0:
            #     test_path = os.path.join(experiment_ckpt_dir, f"test_{iteration}")
            #     os.makedirs(test_path, exist_ok=True)
            #     test_mertic_dict = test_fn(model, device, smplx_model, iteration, fgd_fn, srgr_fn, bc_fn, l1div_fn, cfg.data.test_meta_paths, test_path, cfg)
            #     for k, v in test_mertic_dict.items():
            #         if k not in test_best or v < test_best[k]["value"]:
            #             test_best[k] = {"value": v, "iteration": iteration}  
            #         print(f"Test [{iteration}] - {k}: {v:.6f} (best: {test_best[k]['value']:.6f} at {test_best[k]['iteration']})")
            #         # writer.add_scalar(f"test/{k}", v, iteration)
            #         wandb.log({f"test/{k}": v}, step=iteration)
            #     video_for_log = []
            #     video_res_path = os.path.join(test_path, f"retrieved_motions_{iteration}")
            #     for mp4_file in os.listdir(video_res_path):
            #         if mp4_file.endswith(".mp4"):
            #             # print(mp4_file)
            #             file_path = os.path.join(video_res_path, mp4_file)
            #             log_video = wandb.Video(file_path, caption=f"{iteration:06d}-{mp4_file}", format="mp4")
            #             video_for_log.append(log_video)
            #     wandb.log(
            #       {"test/videos": video_for_log},
            #       step=iteration
            #     )
                # visualize_fn(test_path)
            iteration += 1

    if local_rank == 0:
        writer.close()
    torch.distributed.destroy_process_group()


def init_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_fn(test_path, **kwargs):
    with open(test_path, 'r') as f:
        test_json = json.load(f)
    # load top10_indices from json
    selected_video_path_list = []
    # load video list from json
    with open(test_path, 'r') as f:
        video_list = json.load(f)["video_candidates"]

    for idx, data in enumerate(test_json.items()):
        top10_indices_path = os.path.join(test_path, f"audio_{idx}_retri_top10.json")
        with open(top10_indices_path, 'r') as f:
            top10_indices = json.load(f)["top10_indices"]
        selected_video_path_list.append(video_list[top10_indices[0]])
        # moviepy load and add audio
        video = VideoFileClip(video_list[top10_indices[0]])
        audio = AudioFileClip(data["audio_path"])
        video = video.set_audio(audio)
        video.write_videofile(f"audio_{idx}_retri_top1.mp4")
        video.close()


def prepare_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        config = OmegaConf.load(args.config)
        # config.wandb_project = args.config.split("-")[1]
        config.exp_name = args.config.split("/")[-1][:-5]
    else:
        raise ValueError("Unsupported config file format. Only .yaml files are allowed.")

    if args.debug:
        config.wandb_project = "debug"
        config.exp_name = "debug"
        config.solver.max_train_steps = 4

    if args.overrides:
        for arg in args.overrides:
            key, value = arg.split('=')
            try:
                value = eval(value)
            except:
                pass
            if key in config:
                config[key] = value
            else:
                raise ValueError(f"Key {key} not found in config.")
    
    os.environ["WANDB_API_KEY"] = config.wandb_key

    save_dir = os.path.join(config.output_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)

    config_path = os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith(".py"):
                full_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_file_path, current_dir)
                dest_path = os.path.join(sanity_check_dir, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(full_file_path, dest_path)
    return config


if __name__ == "__main__":
    config = prepare_all()
    main(config)