import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader

from scipy.stats import percentileofscore

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

    df_conve = pd.read_csv(os.path.join(RESULTS_DIR, f"conve_{dataset}_trip_preds_{args.split}.csv"))
    df_tucker = pd.read_csv(os.path.join(RESULTS_DIR, f"tucker_{dataset}_trip_preds_{args.split}.csv"))
    df_nbf = pd.read_csv(os.path.join(RESULTS_DIR, f"nbfnet_{dataset}_trip_preds_{args.split}.csv"))
    df_nbf = nbf_ids_to_ours(data, df_nbf)

    df_conve = df_conve.sort_values(by=['head', 'rel', 'tail'])
    df_tucker = df_tucker.sort_values(by=['head', 'rel', 'tail'])
    df_nbf = df_nbf.sort_values(by=['head', 'rel', 'tail'])

    anyburl_results = read_anyburl_results(data)  # Already sorted and numpy

    models = {
        "NBFNet": df_nbf['hits@10'].to_numpy(),
        "ConvE": df_conve['hits@10'].to_numpy(),
        "TuckER": df_tucker['hits@10'].to_numpy(),
        "AnyBURL": anyburl_results
    }

    return models 



def plot_prob_dists_by_ppr(args):
    """
    Plot the distribution of positive and negative probabilities

    Create a separate plot for different Pos-PPR ranges!
    """
    dataset_name = args.dataset_name

    with open(os.path.join(METRIC_DIR, f'{dataset_name}_probs_by_ppr_range.pkl'), 'rb') as handle:
        data = pickle.load(handle)

    pos_ppr = data['all_pos_pprs']
    pos_probs = data['all_pos_probs']
    corresponding_ppr = data['corresponding_pos_ppr']
    neg_probs = data['sampled_neg_probs']

    all_scores_bin, all_type_bins, all_ppr_bin = [], [], []
    ppr_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]

    ### Make a separate plot for each bin
    # NOTE: Bin only refers to ppr of positive sample
    for pr in ppr_bins:
        pos_scores_bin = pos_probs[((pos_ppr >= pr[0]) & (pos_ppr < pr[1]))]
        neg_scores_bin = neg_probs[((corresponding_ppr >= pr[0]) & (corresponding_ppr < pr[1]))]

        scores_bin = np.concatenate([pos_scores_bin, neg_scores_bin]).tolist()
        type_bin = ["pos"] * len(pos_scores_bin) + ['neg'] * len(neg_scores_bin)
        ppr_bin = [pr] * len(type_bin)

        all_scores_bin.extend(scores_bin)
        all_type_bins.extend(type_bin)
        all_ppr_bin.extend(ppr_bin)
    
    df = pd.DataFrame({"Model Probability": all_scores_bin, "Type": all_type_bins, "Pos PPR Range": all_ppr_bin})

    sns.set_theme("talk")

    g = sns.displot(df, x="Model Probability", hue="Type", kind="hist", col="Pos PPR Range", 
                    stat="probability", common_norm=False, common_bins=False, bins=10, legend=True) #, aspect=1)
    g.figure.suptitle(f'{dataset_name} - NBFNet')

    plt.show()


def plot_prob_dists_by_metric(args):
    """
    Plot the distribution of positive and negative probabilities

    Create a separate plot for different Pos-PPR ranges!
    """
    metric = args.metric
    dataset_name = args.dataset_name

    with open(os.path.join(METRIC_DIR, f'{dataset_name}_pos_neg_metrics.pkl'), 'rb') as handle:
        data = pickle.load(handle)

    neg_probs = np.array(data['neg']['prob'])
    neg_metric = np.array(data['neg'][metric])
    neg_sp = np.array(data['neg']['sp'])

    metric_to_idx = {"sp": 0, "head_deg": 1, "tail_deg": 2, "ppr": 3}
    all_corresponding_metrics = np.array(data['neg']['pos_metrics'])
    pos_metric = all_corresponding_metrics[:, metric_to_idx[metric]]
    pos_probs = all_corresponding_metrics[:, -1]

    if metric == "sp":
        mbins = [(6, 100), (4, 6), (2, 4), (1, 2)]
    elif "deg" in metric:
        mbins = [(1, 5), (5, 10), (10, 100), (100, 1000000)]
    elif metric == "ppr":
        mbins = [(1e-7, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]


    all_scores_bin, all_type_bins, all_metric_bin = [], [], []

    ### Make a separate plot for each bin
    # NOTE: Bin only refers to metric of positive sample
    for mb in mbins:
        mb_ix = ((pos_metric >= mb[0]) & (pos_metric < mb[1]))
        pos_scores_bin = pos_probs[mb_ix]
        neg_scores_bin = neg_probs[mb_ix]
        neg_sp_bin = neg_sp[mb_ix]

        print(f"\n{mb}\n--------")
        print("Pos/Neg Count:", len(pos_scores_bin) / 10, len(pos_scores_bin))
        print("Mean Pos Prob:", pos_scores_bin.mean())
        print("Mean Neg Prob:", neg_scores_bin.mean())
        print("Mean Neg SP:", neg_sp_bin.mean())

        scores_bin = np.concatenate([pos_scores_bin, neg_scores_bin]).tolist()
        type_bin = ["pos"] * len(pos_scores_bin) + ['neg'] * len(neg_scores_bin)
        ppr_bin = [mb] * len(type_bin)

        all_scores_bin.extend(scores_bin)
        all_type_bins.extend(type_bin)
        all_metric_bin.extend(ppr_bin)
    
    df = pd.DataFrame({"Model Probability": all_scores_bin, "Type": all_type_bins, f"Pos {metric} Range": all_metric_bin})

    sns.set_theme("talk")
    g = sns.displot(df, x="Model Probability", hue="Type", kind="hist", col=f"Pos {metric} Range", 
                    stat="probability", common_norm=False, common_bins=False, bins=10, legend=True) #, aspect=1)
    g.figure.suptitle(f'{dataset_name} - NBFNet')

    # plt.show()


def plot_by_prob_by_metric(args, metric):
    """
    Mean positive and negative probability by metric range
    """
    dataset_name = args.dataset_name
    
    if metric.lower() == "ppr":
        metric_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    elif metric.lower() == "sp":
        if args.dataset == "FB15K_237":
            metric_bins = [(2, 3), (3, 4), (4, 1000)]
        else:
            metric_bins = [(1, 3), (3, 5), (5, 7), (7, 1000)]

    with open(os.path.join(METRIC_DIR, f'{dataset_name}_pos_neg_metrics.pkl'), 'rb') as handle:
        pos_neg_data = pickle.load(handle)

    results = {"pos": [], "neg": []}
    for stype in results:
        if stype== "pos":
            probs = np.array(pos_neg_data[stype]['prob']) 
            met_vals = np.array(pos_neg_data[stype][metric.lower()])
        else:
            probs = np.array(pos_neg_data[stype]['prob']['prob']) 
            met_vals = np.array(pos_neg_data[stype]['prob'][metric.lower()])

        for mb in metric_bins:            
            m_ix = (met_vals >= mb[0]) & (met_vals < mb[1])
            results[stype].append(probs[m_ix].mean())

    metric_type = f"Mean Probablity of Sample in {metric.upper()} Range"

    fig, ax = plt.subplots()
    bar_width = .25
    index = np.arange(len(metric_bins)) 
    bin_width_mult = len(metric_bins) / 2 - (len(metric_bins) - 1) * 0.5
    
    ax.bar(index, np.array(results['pos']), bar_width, label="Positive", color=CB_color_cycle[0])
    ax.bar(index+bar_width, np.array(results['neg']), bar_width, label="Negative", color=CB_color_cycle[1])

    if metric.lower() == "ppr":
        metric_bins[-1] = (metric_bins[-1][0], 1)
    else:
        metric_bins[-1] = (metric_bins[-1][0], r"$\infty$")


    dataset_name_title = dataset_name
    if dataset_name == "FB15K_237":
        dataset_name_title = "FB15k-237"
    elif "_v" in dataset_name:
        dd = dataset_name.split("_")
        dataset_name_title = f"{dd[0]} {dd[1]}"

    x_axis_font_size = 12
    x_axis_title = f"{metric.upper()} Score"
    if metric.lower() == "sp":
        x_axis_font_size = 16
        x_axis_title = "Shortest Path Distance"

    plt.title(f"{dataset_name_title}", fontsize=16)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in metric_bins], fontsize=x_axis_font_size) 
    plt.yticks(fontsize=18)
    plt.xlabel(x_axis_title, fontsize=16)
    plt.ylabel(f"Mean Probability", fontsize=16)
    plt.tight_layout()
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.22), prop = {'size': 14})

    ### Save image correctly
    dataset_name = dataset_name.lower().replace("-", '_') if args.version is None else dataset_name  # DON'T ASK!!!
    if not os.path.isdir(os.path.join(IMG_DIR, dataset_name)):
        os.mkdir(os.path.join(IMG_DIR, dataset_name))
    
    fff = os.path.join(IMG_DIR, dataset_name, f"Probabilty_By_Range_{metric.lower().capitalize()}_NBFNet.png")
    plt.savefig(fff, dpi=300, bbox_inches='tight')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="NBFNet")
    parser.add_argument("--metric", type=str, default="ppr")
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

    plot_by_prob_by_metric(args, "PPR")
    plot_by_prob_by_metric(args, "SP")
    # plot_by_prob_by_metric(args, "tail_deg")
    # plot_by_prob_by_metric(args, "head_deg")
    # plot_prob_dists_by_ppr(args)

    # plot_prob_dists_by_metric(args)


if __name__ == "__main__":
    main()
