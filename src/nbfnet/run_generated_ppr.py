import os
import sys
import math
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm

import pickle
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


separator = ">" * 30
line = "-" * 30

@torch.no_grad()
def test(cfg, test_data, filtered_data=None, is_valid=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    rankings = []
    num_negatives = []

    # Load PPR
    is_valid = "_val" if is_valid and cfg['is_inductive'] else ""
    ddir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "ppr", cfg['dataset_name'])
    file_name = f"sparse_adj-{cfg['alpha']}_eps-{cfg['eps']}".replace(".", "") + is_valid + ".pt"
    print(ddir)
    ppr_matrix = torch.load(os.path.join(ddir, file_name))

    for batch in tqdm(test_loader, "Testing"):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        
        t_pred = torch.index_select(ppr_matrix, 0, pos_h_index.cpu()).to_dense().to(device)
        h_pred = torch.index_select(ppr_matrix, 0, pos_t_index.cpu()).to_dense().to(device)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)
        
        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

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
    dataset_name = cfg.dataset["class"]

    cfg['alpha'] = args.alpha
    cfg['eps'] = args.eps
    cfg['is_inductive'] = True
    cfg['dataset_name'] = args.dataset

    # Build valid/test data
    datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data", "generated")

    with open(os.path.join(datadir, f"{args.dataset}.pkl"), 'rb') as handle:
        rawdata = pickle.load(handle)

    edge_index = torch.Tensor([(int(t[0]), int(t[2])) for t in rawdata['inf_graph']]).t().long()
    edge_type = torch.Tensor([int(t[1]) for t in rawdata['inf_graph']]).t().long()

    num_nodes = torch.max(edge_index).item() + 1
    cfg.model.num_relation = torch.max(edge_type).item() + 1

    filtered_data = None
    device = util.get_device(cfg)

    std_ei = [(int(t[0]), int(t[2])) for t in rawdata['test_samples']]
    std_et = [int(t[1]) for t in rawdata['test_samples']]

    target_edge_index = torch.Tensor(std_ei).t().long()
    target_edge_type = torch.Tensor(std_et).t().long()
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                     target_edge_index=target_edge_index, target_edge_type=target_edge_type)

    test_data = test_data.to(device)

    logger.warning(separator)
    logger.warning("Evaluate Test")
    test(cfg, test_data, filtered_data=filtered_data)
