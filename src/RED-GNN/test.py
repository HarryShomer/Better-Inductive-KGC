import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
# from load_data_old import DataLoader
from base_model import BaseModel
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--num-test', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--dropout', type=float, default=0.1)

args = parser.parse_args()

class Options(object):
    pass


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    # torch.cuda.set_device(f"cuda:{args.device}")
    # print('gpu:', gpu)

    loader = DataLoader(args.data_path, args.num_test)
    # loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    opts.lr = args.lr
    opts.lamb = 0.0001
    opts.decay_rate = 0.995
    # For fair comparison with NBFNet
    # Also often causes OOM when > 32 on codex-m and hetionet
    opts.hidden_dim = 32

    opts.attn_dim = 5
    opts.dropout = args.dropout
    opts.act = 'relu'
    opts.n_layer = 5
    opts.n_batch = 64
    epochs = 20 
   
    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)

    model = BaseModel(opts, loader)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(file_dir, "ckpt", dataset)

    print("Loading Saved Model...")
    model.model.load_state_dict(torch.load(os.path.join(save_dir, f"red_gnn_best_{args.seed}.ckpt"))["model_state_dict"])

    _, out_str = model.evaluate()

    print(out_str)


