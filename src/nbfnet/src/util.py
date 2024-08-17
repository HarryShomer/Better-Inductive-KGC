import os
import sys
import ast
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.data import Data

from src import models, datasets
from src.neurallp import NeuralLogicProgramming


logger = logging.getLogger(__file__)



def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument("-d", "--dataset", help="When YAML not used", type=str, default=None)
    parser.add_argument("--checkpoint", help="File with model to load", type=str)

    parser.add_argument("--save_as", help="Name of model file", type=str, default=None)

    ## PPR
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--eps", type=float, default=1e-6)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg, args):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    if args.save_as is None:
        args.save_as = f"Seed-{args.seed}"

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], 
                               args.save_as)

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg, args=None):
    cls = cfg.dataset.pop("class")
    is_new = cfg.dataset.get("new", False)
    
    if is_new:
        dataset = datasets.NewSplit(cfg.dataset.root, cls, num_test=cfg.dataset.num_test)
    elif cls == "FB15k-237":
        dataset = datasets.FB15k237(cfg.dataset.root)
    elif cls == "WN18RR":
        dataset = datasets.WN18RR(cfg.dataset.root)
    elif cls.lower() == "codex-s":
        dataset = datasets.CoDExSmall(cfg.dataset.root)
    elif cls.lower() == "codex-m":
        dataset = datasets.CoDExMedium(cfg.dataset.root)
    elif cls.lower() == "codex-l":
        dataset = datasets.CoDExLarge(cfg.dataset.root)
    elif cls.lower() == "yago310":
        dataset = datasets.YAGO310(cfg.dataset.root)
    elif cls.lower() == "dbpedia100k":
        dataset = datasets.DBpedia100k(cfg.dataset.root)
    elif cls.lower() == "hetionet":
        dataset = datasets.Hetionet(cfg.dataset.root)
    elif cls.startswith("Ind"):
        dataset = datasets.GrailInductiveDataset(name=cls[3:], **cfg.dataset)
    elif cls.lower() == "ilpc":
        dataset = datasets.ILPC2022(**cfg.dataset) 
    elif cls.lower() == "fb-ingram":
        dataset = datasets.FBIngram(**cfg.dataset)
    elif cls.lower() == "wk-ingram":
        dataset = datasets.WKIngram(**cfg.dataset)
    else:
        raise ValueError("Unknown dataset `%s`" % cls)

    if get_rank() == 0:
        logger.warning("%s dataset" % cls)
        logger.warning(f"# Train Node = {dataset[0].num_nodes}")
        logger.warning("#train: %d, #valid: %d, #test: %d" %
                       (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                        dataset[2].target_edge_index.shape[1]))

    return dataset


def build_model(cfg, data, ppr_test=False):
    cls = cfg.model.pop("class")
    if cls == "NBFNet":
        model = models.NBFNet(**cfg.model, ppr_test=ppr_test)
    elif cls == "NeuralLP":
        model = NeuralLogicProgramming(**cfg.model)
    else:
        raise ValueError("Invalid Model")
    
    if "checkpoint" in cfg:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    return model
