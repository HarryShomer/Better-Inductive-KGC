import os
import sys
import math
import pprint
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import tasks, util


separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(1, cfg.train.num_epoch+1):
        epoch=i
        parallel_model.train()

        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Epoch %d begin" % epoch)

        losses = []
        sampler.set_epoch(epoch)

        for batch in tqdm(train_loader, f"Epoch {i}"):
            batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                            strict=cfg.task.strict_negative)
            pred = parallel_model(train_data, batch)
            target = torch.zeros_like(pred)
            target[:, 0] = 1

            # Label Smoothing
            ls = cfg.train.get("label_smooth", 0)
            if ls != 0.0:
                target = (1.0 - cfg.train.label_smooth)*target + (1.0 / train_data.num_nodes)

            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            neg_weight = torch.ones_like(pred)
            if cfg.task.adversarial_temperature > 0:
                with torch.no_grad():
                    neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
            else:
                neg_weight[:, 1:] = 1 / cfg.task.num_negative
            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                logger.warning(separator)
                logger.warning("binary cross entropy: %g" % loss)
            losses.append(loss.item())
            batch_id += 1

        if util.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average binary cross entropy: %g" % avg_loss)

        if i % 2 == 0 and i != 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        # Eval every 2 epochs
        if i % 2 == 0 and i != 0: 
            logger.warning(separator)
            logger.warning("Evaluate on valid")
            result = test(cfg, model, valid_data, filtered_data=filtered_data)
            if result > best_result:
                best_result = result
                best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None, split=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []

    for batch in tqdm(test_loader, "Testing"):
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
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

    if args.checkpoint is None:
        working_dir = util.create_working_directory(cfg, args)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    dataset_name = cfg.dataset["class"]
    
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

    dataset = util.build_dataset(cfg, dataset_args)
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg, dataset)

    device = util.get_device(cfg)
    model = model.to(device)

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

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        logger.warning(separator)
        logger.warning("Evaluate on valid")
        test(cfg, model, valid_data, filtered_data=filtered_data, split="valid")
        logger.warning(separator)
        logger.warning("Evaluate on test")

        for i in range(2, len(dataset)):
            print(f">>> Test Graph {i-2}")
            test_graph = dataset[i].to(device)
            test(cfg, model, test_graph, filtered_data=filtered_data)

        exit()

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")

    for i in range(2, len(dataset)):
        print(f">>> Test Graph {i-2}")
        test_graph = dataset[i].to(device)
        test(cfg, model, test_graph, filtered_data=filtered_data)
