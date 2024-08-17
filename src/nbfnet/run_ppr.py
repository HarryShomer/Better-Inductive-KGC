import os
import sys
import math
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import tasks, util

sys.path.append("..") 
from calc_ppr_matrices import *

separator = ">" * 30
line = "-" * 30


def get_ppr(cfg, data):
    """
    Get PPR matrix.

    If it exists, load it in

    Otherwise create and save it
    """
    ddir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "ppr", cfg['dataset_name'])
    file_name = f"sparse_adj-{cfg['alpha']}_eps-{cfg['eps']}".replace(".", "") + ".pt"
    
    if os.path.isfile(os.path.join(ddir, file_name)):
        print("Loading PPR...")
        ppr_matrix = torch.load(os.path.join(ddir, file_name))
    else:
        alpha, eps = cfg['alpha'], cfg['eps'] 
        neighbors, neighbor_weights = get_ppr_matrix(data.edge_index, data.num_nodes, alpha, eps)
        ppr_matrix = create_sparse_ppr_matrix(neighbors, neighbor_weights)
        save_results(cfg['dataset_name'], ppr_matrix, alpha, eps, val = False)
    
    return ppr_matrix


@torch.no_grad()
def test(cfg, test_data, filtered_data=None, is_valid=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    rankings = []
    num_negatives = []
    pos_preds, neg_preds = [], []
    top1_neg_preds, top10_neg_preds = [], []

    # Load PPR
    ppr_matrix = get_ppr(cfg, test_data)

    for batch in tqdm(test_loader, "Testing"):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        
        t_pred = torch.index_select(ppr_matrix, 0, pos_h_index.cpu()).to_dense().to(device)
        h_pred = torch.index_select(ppr_matrix, 0, pos_t_index.cpu()).to_dense().to(device)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        
        pos_t_pred = t_pred.gather(-1, pos_t_index.unsqueeze(-1))
        pos_h_pred = h_pred.gather(-1, pos_h_index.unsqueeze(-1))
        pos_preds += [pos_t_pred, pos_h_pred]

        # neg_preds += [t_pred[t_mask], h_pred[h_mask]]

        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)
        
        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        t_pred[~t_mask] = 0
        h_pred[~h_mask] = 0
        t_topk = torch.topk(t_pred, 10, dim=-1).values.mean(axis=-1)
        h_topk = torch.topk(h_pred, 10, dim=-1).values.mean(axis=-1)
        top10_neg_preds += [t_topk, h_topk]

    pos_preds = torch.cat(pos_preds).squeeze(-1)
    # neg_preds = torch.cat(neg_preds)
    # top1_neg_preds = torch.cat(top1_neg_preds) + cfg['alpha']  # Avoid div/0
    top10_neg_preds = torch.cat(top10_neg_preds) #+ cfg['alpha']


    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    dataset_name = cfg.dataset["class"]

    dataset = util.build_dataset(cfg, args)
    cfg.model.num_relation = dataset.num_relations

    ddd = dataset_name.lower()
    dataset_args = None
    if ddd.startswith("ind") or "ilpc" in ddd or "ingram" in ddd or ddd.startswith("wk"):
        is_inductive = True 
        cfg['dataset_name'] = f"{dataset_name}_{cfg.dataset['version']}"
    elif cfg.dataset.get("new"):
        is_inductive = True 
        cfg['dataset_name'] = dataset_name
        dataset_args = args
    else:
        is_inductive = False
        cfg['dataset_name'] = dataset_name

    device = util.get_device(cfg)
    train_data, valid_data = dataset[0], dataset[1]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
        filtered_data = filtered_data.to(device)

    #################################
    if dataset_name.lower().startswith("ind"):
        dataset_name = dataset_name[3:]

    cfg['alpha'] = args.alpha
    cfg['eps'] = args.eps
    cfg['is_inductive'] = is_inductive
    #################################

    # logger.warning("Evaluate on valid")
    # test(cfg, valid_data, filtered_data=filtered_data, is_valid=True)
    logger.warning(separator)
    logger.warning("Evaluate on test")

    # Control for (E, R) datasets
    if len(dataset) == 3:
        test_data = dataset[2].to(device)
        test(cfg, test_data, filtered_data=filtered_data)
    else:
        for i in range(2, len(dataset)):
            print(f">>> Test Graph {i-2}")
            cfg['dataset_name'] = dataset_name + f"_Test_{i-2}"
            test_graph = dataset[i].to(device)
            # print(test_graph.num_nodes,test_graph.num_relations)
            # print(test_graph.edge_index.shape, test_graph.edge_type.shape)
            # print(test_graph.target_edge_index.shape, test_graph.target_edge_type.shape)
            test(cfg, test_graph, filtered_data=filtered_data)


