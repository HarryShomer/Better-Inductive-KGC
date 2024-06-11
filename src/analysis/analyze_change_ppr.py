import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader

import seaborn as sns

from kgpy import datasets
from util import *


def get_inductive_results(dataset, split, sprange="1-100"):
    """
    Returns as dictionary
    """
    df_sp = pd.read_csv(os.path.join(RESULTS_DIR, f"nbfnet_{dataset.lower()}_SP-{sprange}_preds_{split}.csv"))
    df_deg = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset}_degree_{split}.csv"))

    df_sp = df_sp.sort_values(by=['head', 'rel', 'tail'])
    df_deg = df_deg.sort_values(by=['head', 'rel', 'tail'])

    data = {
        "hits@10": df_sp['hits@10'].to_numpy(),
        "sp": df_sp['sp'].to_numpy(),
        "tail_degree": df_deg['tail_degree'].to_numpy(),
        "head_degree": df_deg['head_degree'].to_numpy(),
        "tails": df_deg['tail'].to_numpy()
    }

    return data 



def get_transductive_results(args):
    """
    Returns as dictionary
    """
    data = getattr(datasets, args.dataset.upper().replace("-", "_"))(inverse=True)

    dataset = data.dataset_name.lower().replace("-", "_")

    # df_conve = pd.read_csv(os.path.join(RESULTS_DIR, f"conve_{dataset}_trip_preds_{args.split}.csv"))
    # df_tucker = pd.read_csv(os.path.join(RESULTS_DIR, f"tucker_{dataset}_trip_preds_{args.split}.csv"))
    df_nbf = pd.read_csv(os.path.join(RESULTS_DIR, f"nbfnet_{dataset}_trip_preds_{args.split}.csv"))
    # df_nbf = nbf_ids_to_ours(data, df_nbf)

    # df_conve = df_conve.sort_values(by=['head', 'rel', 'tail'])
    # df_tucker = df_tucker.sort_values(by=['head', 'rel', 'tail'])
    # df_nbf = df_nbf.sort_values(by=['head', 'rel', 'tail'])

    # anyburl_results = read_anyburl_results(data)  # Already sorted and numpy

    # models = {
    #     "NBFNet": df_nbf['hits@10'].to_numpy(),
    #     "ConvE": df_conve['hits@10'].to_numpy(),
    #     "TuckER": df_tucker['hits@10'].to_numpy(),
    #     "AnyBURL": anyburl_results
    # }

    return df_nbf 



def read_pos_probs(args):
    """
    Samples are in same order as nbfnet probs
    """
    dataset = args.dataset
    if args.version is not None:
        dataset = "ind" + args.dataset + args.version

    # Original
    samples = torch.load(os.path.join(PRED_DIR, f"nbfnet_{dataset.lower()}_samples.pt"))
    pos_probs = torch.load(os.path.join(PRED_DIR, f"nbfnet_{dataset.lower()}_pos_preds.pt"))
    pos_probs = torch.sigmoid(pos_probs).squeeze(-1).tolist()

    original_probs = {}
    for s, p in zip(samples.t().tolist(), pos_probs):
        original_probs[tuple(s)] = p

    # New after ppr modify
    ddir = os.path.join(METRIC_DIR, "..", "modify_ppr", f"{args.dataset}_{args.version}")
    with open(os.path.join(ddir, f"{args.dataset}_{args.version}_low_trips.pkl"), 'rb') as handle:
        low_data = pickle.load(handle)
    with open(os.path.join(ddir, f"{args.dataset}_{args.version}_high_trips.pkl"), 'rb') as handle:
        high_data = pickle.load(handle)
    
    # TODO: Add old hits@10
    # Merge data
    # Format: [old_ppr, new_ppr, new hits@10, new_pred, old_pred]
    low_data_list, high_data_list = [], []
    for k in low_data:
        low_data_list.append(low_data[k] + [original_probs[k]])
    for k in high_data:
        high_data_list.append(high_data[k] + [original_probs[k]])

    # # Transductive need to be converted to our IDs
    # if args.version is None:
    #     samples = nbf_ids_to_ours(samples, dataset)

    return low_data_list, high_data_list



def plot_by_prob_by_ppr(args):
    """
    Mean positive and negative probability by ppr score range
    """
    dataset_name = args.dataset_name
    metric_type = "Mean Probablity of Sample in PPR Range"

    with open(os.path.join(METRIC_DIR, f'{dataset_name}_probs_by_ppr_range.pkl'), 'rb') as handle:
        data = pickle.load(handle)

    ppr_bins = list(data['pos']['probs'].keys())
    pos_probs = list(data['pos']['probs'].values())
    neg_probs = list(data['neg']['probs'].values())

    fig, ax = plt.subplots()
    bar_width = .25
    index = np.arange(len(ppr_bins)) 
    bin_width_mult = len(ppr_bins) / 2  - 2
    
    ax.bar(index, np.array(pos_probs), bar_width, label="Positive", color=CB_color_cycle[0])
    ax.bar(index+bar_width, np.array(neg_probs), bar_width, label="Negative", color=CB_color_cycle[1])

    plt.title(f"{dataset_name}: NBFNet", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in ppr_bins], fontsize=10) 
    plt.yticks(fontsize=18)
    plt.xlabel("PPR Score", fontsize=16)
    plt.ylabel(f"Mean Probability", fontsize=16)
    plt.tight_layout()
    ax.legend(title=metric_type, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.26), prop = {'size': 12})

    ### Save image correctly
    dataset_name = dataset_name.lower().replace("-", '_') if args.version is None else dataset_name  # DON'T ASK!!!
    if not os.path.isdir(os.path.join(IMG_DIR, dataset_name)):
        os.mkdir(os.path.join(IMG_DIR, dataset_name))
    
    fff = os.path.join(IMG_DIR, dataset_name, f"Probabilty_By_Range_PPR_NBFNet.png")
    plt.savefig(fff, dpi=300, bbox_inches='tight')



def plot_modify_ppr_prob(dataset, trip_info, trip_type="low"):
    """
    X-axis=Change in PPR
    Y-axis=Change in Prob
    """
    ppr_diff  = np.array([(t[1] - t[0]) for t in trip_info])
    prob_diff = np.array([(t[-2] - t[-1])  for t in trip_info])

    plt.figure()
    # ax.bar(index, np.array(mean_prob_by_bin), bar_width)
    plt.scatter(ppr_diff, prob_diff)
    
    plt.title(f"{dataset} - NBFNet ({trip_type.capitalize()} PPR)", fontsize=18)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16)
    plt.xlabel("PPR Difference", fontsize=16)
    plt.ylabel("Prob Difference", fontsize=16)
    plt.tight_layout()

    fff = os.path.join(IMG_DIR, dataset, f"PPR-Diff_vs_Prob-Diff_{trip_type}-samples.png")
    plt.savefig(fff, dpi=300, bbox_inches='tight')





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="NBFNet")
    parser.add_argument("--eps",type=float, default=1e-6)
    args = parser.parse_args()

    ### Transductive or inductive
    if args.version is None:
        args.eps = 1e-6
        args.dataset_name = args.dataset
        results = get_transductive_results(args)[args.model]
    else:
        args.eps = 1e-7
        args.dataset_name = f"{args.dataset}_{args.version}"
        results = get_inductive_results(args.dataset_name, args.split)['hits@10']

    # plot_by_prob_by_ppr(args)
        
    low_ppr_trips, high_ppr_trips = read_pos_probs(args)

    # print(low_ppr_trips)

    # low_change = (np.mean([t[-2] / t[-1] for t in low_ppr_trips]) - 1) * 100
    # high_change = (np.mean([t[-2] / t[-1] for t in high_ppr_trips]) - 1) * 100
    # print("Mean % change of prob for low:", round(low_change, 2))
    # print("Mean % change of prob for high:", round(high_change, 2))

    plot_modify_ppr_prob(args.dataset_name, low_ppr_trips)
    plot_modify_ppr_prob(args.dataset_name, high_ppr_trips, "high")


if __name__ == "__main__":
    main()
