import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.utils import degree

from scipy.stats import percentileofscore

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from util import *


def get_data(dataset, split, sprange="1-100", eps=1e-7, alpha=0.15):
    """
    Returns as dictionary
    """
    df_sp = pd.read_csv(os.path.join(RESULTS_DIR, f"nbfnet_{dataset.lower()}_SP-{sprange}_preds_{split}.csv"))
    df_sp = df_sp.sort_values(by=['head', 'rel', 'tail'])

    df_deg = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset}_degree_{split}.csv"))
    df_deg = df_deg.sort_values(by=['head', 'rel', 'tail'])

    ddir = os.path.join(PPR_DIR, dataset)
    df_ppr = pd.read_csv(os.path.join(ddir, f"test_ppr_eps-{str(eps).replace('.', '')}_alpha-{str(alpha)}.csv")) 
    df_ppr = df_ppr.sort_values(by=['head', 'rel', 'tail'])

    metric_df = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset}_resistance.csv")) 
    metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])

    data = {
        "hits@10": df_sp['hits@10'].to_numpy(),

        "sp": df_sp['sp'].to_numpy(),        
        "tail_degree": df_deg['tail_degree'].to_numpy(),
        "head_degree": df_deg['head_degree'].to_numpy(),
        "walks": read_walks(dataset),
        "ppr": df_ppr['ppr'].to_numpy(),
        "ppr_tgt": df_ppr['ppr_tgt'].to_numpy(),
        "resistance": metric_df["resistance"].to_numpy()
    }

    return data 


def metric1_vs_metric2(dataset_name, version, metric1, metric2, eps=1e-7, alpha=0.15):
    """
    Correlation between two metrics
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, "test")

    metric1_vals = data[metric1]
    metric2_vals = data[metric2]
        
    reg = LinearRegression().fit(metric1_vals.reshape(-1, 1), metric2_vals)
    textstr=f"R^2 = {round(reg.score(metric1_vals.reshape(-1, 1), metric2_vals), 3)}"
    print(textstr)

    fig, ax = plt.subplots()
    plt.scatter(metric1_vals, metric2_vals)
    plt.plot(np.unique(metric1_vals), np.poly1d(np.polyfit(metric1_vals, metric2_vals, 1))(np.unique(metric1_vals)), color="red")

    # these are matplotlib.patch.Patch properties
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.75, textstr, transform=ax.transAxes, fontsize=14, bbox=props)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14) 
    plt.title(f"{dataset_name}: {metric1} vs. {metric2}", fontsize=15)
    plt.xlabel(metric1, fontsize=15)
    plt.ylabel(metric2, fontsize=15)

    ax.set_ylim([-5, metric2_vals.max() + 5])

    if not os.path.isdir(os.path.join(IMG_DIR, dataset_name)):
        os.mkdir(os.path.join(IMG_DIR, dataset_name))
    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"{metric1}_vs_{metric2}.png"), dpi=300, bbox_inches='tight')


def performance_by_metric(dataset_name, version, metric, split="test"):
    """
    For different ranges of the metric we get the performance of each method

    metric = ['SP', 'head_degree', 'rel_degree']
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, split)

    model_results = data['hits@10']
    metric_vals = data[metric.lower()]

    if metric == "SP":
        metric_name = "Shortest Path"
        if "wn" in dataset_name:
            bins = [(1, 2), (2, 4), (4, 6), (6, 9), (9, 1000)]
        else:
            bins = [(2, 3), (3, 4), (4, 1000)]
    elif "degree" in metric:
        metric_name = metric.capitalize()
        bins = [(1, 3), (3, 10), (10, 1000000)]
    elif "walks" in metric:
        metric_name = "Walks Percentile"
        metric_vals = np.array([percentileofscore(metric_vals, i, kind='strict') for i in metric_vals])
        bins = [(i * 10, (i+1) * 10) for i in range(10)]

    ### Categorize results for each model by bin
    model_perf_bins = []
    for b in bins:
        trip_ix = (metric_vals >= b[0]) & (metric_vals < b[1])
        print(f"# Samples {b}:", sum(trip_ix))
        model_perf_bins.append(model_results[trip_ix].mean())
            
    fig, ax = plt.subplots()
    bar_width = 0.2
    index = np.arange(len(bins))
    bin_width_mult = len(bins) / 2 - (len(bins) - 2) * 0.5

    bbb = ax.bar(index, np.array(model_perf_bins), bar_width)

    bins[-1] = (bins[-1][0], r"$\infty$")

    plt.title(dataset_name, fontsize=18)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in bins], fontsize=10, rotation=15) 
    plt.yticks(fontsize=18)
    plt.xlabel(metric_name, fontsize=18)
    plt.ylabel(f"Hits@10", fontsize=18)
    plt.tight_layout()
    # ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2),  prop = {'size': 14})

    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Perf_by_{metric}.png"), dpi=300, bbox_inches='tight')


def performance_by_SP_metric(dataset_name, version=None, metric="tail_degree", split="test", eps=None, alpha=0.15):
    """
    Queso!
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, split)

    sp_bins = [(1, 2), (2, 4), (4, 6), (6, 1000)]

    sp_vals = data['sp']
    hits = data['hits@10']
    metric_vals = data[metric.lower()]

    if metric == "ppr":
        metric_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    elif metric == "resistance":
        metric_bins = [(0, 0.1), (0.1, 0.5), (0.5, 1), (1,  1000)]
    elif metric == "walks":
        metric_vals = np.array([percentileofscore(metric_vals, i, "strict") for i in metric_vals])
        if "fb" in dataset_name.lower(): 
            metric_bins = [(0, 10), (10, 33), (33, 66), (66, 100)] 
        else: 
            [(0, 33), (33, 66), (66, 100)]
    # elif metric == "spd_walks":
    #     walk_vals = read_walks(dataset_name, cum=False)
    #     metric_vals = get_metric_for_spd(sp_vals, walk_vals)
    #     metric_vals = np.array([percentileofscore(metric_vals, i,  "strict") for i in metric_vals])
        
    #     if "fb" in dataset_name.lower(): 
    #         metric_bins = [(0, 10), (10, 33), (33, 66), (66, 100)] 
    #     else: 
    #         metric_bins = [(0, 33), (33, 66), (66, 100)]
    else:
        metric_bins = [(1, 3), (3, 10), (10, 1000000)]
    
    model_perf_bins = defaultdict(list)
    for ix, sp_bin in enumerate(sp_bins):
        sp_ix = (sp_vals >= sp_bin[0]) & (sp_vals < sp_bin[1])

        for d_bin in metric_bins:
            deg_ix = (metric_vals >= d_bin[0]) & (metric_vals < d_bin[1])
            both_ix = sp_ix * deg_ix
            print(">>>", sp_bin, "/", d_bin, "=", both_ix.sum().item())
            model_perf_bins[d_bin].append(np.nan_to_num(hits[both_ix].mean()))
    
    
    fig, ax = plt.subplots()
    bar_width = 0.15
    index = np.arange(len(sp_bins)) 
    bin_width_mult = len(sp_bins) / 2 - 0.5
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
    
    sp_bins[-1] = (sp_bins[-1][0], r"$\infty$")

    plt.title(f"{dataset_name}: NBFNet", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in sp_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("SP", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=metric.capitalize(), frameon=False, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})

    if not os.path.isdir(os.path.join(IMG_DIR, dataset_name)):
        os.mkdir(os.path.join(IMG_DIR, dataset_name))
    mmm = '-'.join(metric.split()) if "Degree" in metric else metric
    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Control_SP_{mmm}.png"), dpi=300, bbox_inches='tight')



def corr_analysis(dataset_name, version=None,  split="test", eps=1e-7, alpha=0.15):
    """
    Queso!
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, split)

    sp_vals = data['sp']
    model_results = data['hits@10']

    ppr_vals = data['ppr']
    ppr_tgt_vals = data['ppr_tgt']
    tail_deg_vals = data['tail_degree']
    head_deg_vals = data['head_degree']
    walk_vals = data['walks']

    sp_not100 = sp_vals != 100
    sp_vals = sp_vals[sp_not100]
    head_deg_vals = head_deg_vals[sp_not100]
    tail_deg_vals = tail_deg_vals[sp_not100]
    walk_vals = walk_vals[sp_not100]
    ppr_vals = ppr_vals[sp_not100]
    preds = model_results[sp_not100]

    # R^2 = Via Linear Regressions
    X = []
    for i, j, k, l, m, n in zip(sp_vals, head_deg_vals, tail_deg_vals, walk_vals, ppr_vals, ppr_tgt_vals):
        X.append([i, j, k, l])

    X = np.array(X)
    X = StandardScaler().fit_transform(X)
    reg = LogisticRegression(penalty="none").fit(X, preds)
    print("R =", np.sqrt(round(reg.score(X, preds), 4)))
    print(reg.coef_)



def performance_by_Walks_metric(dataset_name, version=None, metric="tail_degree", split="test", 
                                eps=None, alpha=1e-7):
    """
    Queso!
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, split)

    hits = data['hits@10']
    metric_vals = data[metric.lower()]

    if metric == "ppr":
        metric_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    else:
        metric_bins = [(1, 3), (3, 10), (10, 1000000)]
    
    ### Walks
    walk_bins = [(0, 10), (10, 30), (30, 60), (60, 100)]
    walks_vals = data['walks']
    walks_vals = np.array([percentileofscore(walks_vals, i, kind='strict') for i in walks_vals])

    model_perf_bins = defaultdict(list)
    for ix, w_bin in enumerate(walk_bins):
        w_ix = (walks_vals >= w_bin[0]) & (walks_vals < w_bin[1])

        for d_bin in metric_bins:
            deg_ix = (metric_vals >= d_bin[0]) & (metric_vals < d_bin[1])
            both_ix = w_ix * deg_ix
            print(">>>", w_bin, "/", d_bin, "=", both_ix.sum().item())
            model_perf_bins[d_bin].append(hits[both_ix].mean())
    
    fig, ax = plt.subplots()
    bar_width = 0.15
    index = np.arange(len(walk_bins)) 
    bin_width_mult = len(metric_bins) / 2 - 0.5
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
    
    # walk_bins[-1] = (walk_bins[-1][0], r"$\infty$")

    plt.title(f"{dataset_name}: NBFNet", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in walk_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("Walks Percentile", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()

    metric = metric.capitalize()
    ax.legend(title=metric, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})

    if not os.path.isdir(os.path.join(IMG_DIR, dataset_name)):
        os.mkdir(os.path.join(IMG_DIR, dataset_name))
    mmm = '-'.join(metric.split()) if "Degree" in metric else metric
    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Control_Raw_Walks_{mmm}_NBFNet.png"), dpi=300, bbox_inches='tight')


def analyze_uncommon(dataset_name, version, split="test", eps=1e-5, alpha=0.15):
    """
    Analyze:
        1. Low PPR samples that do well
        2. High PPR samples that do well 
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, split)

    sp_vals = data['sp']
    model_results = data['hits@10']
    ppr_vals = data['ppr']
    tail_deg_vals = data['tail_degree']
    head_deg_vals = data['head_degree']
    walks_vals = data['walks']
    # unq_vals = getz_unique_vals(dataset_name)


    pbins = [(1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    for pbin in pbins:

        low_ppr_wrong_ix = (ppr_vals >= pbin[0]) & (ppr_vals < pbin[1]) & (model_results == 0)
        low_ppr_right_ix = (ppr_vals >= pbin[0]) & (ppr_vals < pbin[1]) & (model_results == 1) 

        if low_ppr_wrong_ix.sum() == 0 or low_ppr_right_ix.sum() == 0:
            continue

        print(f"\n{pbin} PPR Mean Stats (Wrong/Correct):\n-----------------------------")
        print("# Samples =", low_ppr_wrong_ix.sum(), low_ppr_right_ix.sum())
        print("  PPR:", ppr_vals[low_ppr_wrong_ix].mean(), ppr_vals[low_ppr_right_ix].mean())
        print("  SP:", sp_vals[low_ppr_wrong_ix].mean(), sp_vals[low_ppr_right_ix].mean())
        print("  Walks:", walks_vals[low_ppr_wrong_ix].mean(), walks_vals[low_ppr_right_ix].mean())
        print("  Tail Degree:", tail_deg_vals[low_ppr_wrong_ix].mean(), tail_deg_vals[low_ppr_right_ix].mean())
        print("  Head Degree:", head_deg_vals[low_ppr_wrong_ix].mean(), head_deg_vals[low_ppr_right_ix].mean())

        # Get # unique vals by hop
        unq_vals_wrong_ix = unq_vals[low_ppr_wrong_ix] 
        unq_vals_right_ix = unq_vals[low_ppr_right_ix] 
        for l in range(unq_vals_wrong_ix.shape[1]):
            print(f"  Unique l={l+1}:", unq_vals_wrong_ix[:, l].mean(), unq_vals_right_ix[:, l].mean())



def performance_by_walks_per_metric(dataset_name, version=None, metric="tail_degree", 
                                    split="test", eps=None, alpha=1e-7):
    """
    Queso!
    """
    dataset_name = f"{dataset_name}_{version}"
    data = get_data(dataset_name, split)

    hits = data['hits@10'] 
    sp_vals = data['sp']
    sp_bins = [(1, 2), (2, 4), (4, 6), (6, 1000)]

    ### PPR
    ddir = os.path.join(PPR_DIR, dataset_name)
    metric_df = pd.read_csv(os.path.join(ddir, f"test_ppr_eps-{str(eps).replace('.', '')}_alpha-{str(alpha)}.csv")) 
    metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])
    sp_vals = data['ppr']
    sp_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]

    ### Walks
    walk_vals = read_walks(dataset_name, cum=False)

    # Unique
    unique_df = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset_name}_Unique.csv"))
    unique_df = unique_df.sort_values(by=['head', 'rel', 'tail'])

    if "wn" in dataset_name.lower():
        unq_bins = [(0, 33), (33, 66), (66, 100)]
    else:
        unq_bins = [(0, 10), (10, 33), (33, 66), (66, 100)]

    unq_l_vals = []
    walk_l_vals = []
    for ix, unq in enumerate(unique_df.to_dict("records")):
        isp = int(data['sp'][ix])
        if isp <= 6:
            # unq_l_vals.append(unq[f'unique_{isp}'])
            walk_l_vals.append(walk_vals[ix][isp-1])
        else:
            # unq_l_vals.append(1)
            walk_l_vals.append(0)
        unq_l_vals.append(unq[f'unique_1'])
    unq_l_vals, walk_l_vals = np.array(unq_l_vals), np.array(walk_l_vals)

    if metric == "unique":
        walks_per_unq = unq_l_vals # walk_l_vals / unq_l_vals
    elif "degree" in metric:
        walks_per_unq = walk_l_vals / data["tail_degree"]

    # walks_per_unq = []
    # for ix, unq in enumerate(unique_df.to_dict("records")):
    #     isp = int(data['sp'][ix])
    #     if isp == 1:
    #         walks_per_unq.append(unq[f'unique_1'])
    #     elif isp <= 6:
    #         unql = [unq[f'unique_{i}'] /  walk_vals[ix][i-1] for i in range(2, isp+1)]
    #         walks_per_unq.append(min(unql))
    #     else:
    #         walks_per_unq.append(0)
    # walks_per_unq = np.array(walks_per_unq)

    unq_bins = [(1, 3), (3, 10), (10,  10000)]


    # walks_per_unq = np.array([percentileofscore(walks_per_unq, i, kind='strict') for i in walks_per_unq])

    model_perf_bins = defaultdict(list)
    for ix, m_bin in enumerate(sp_bins):
        m_ix = (sp_vals >= m_bin[0]) & (sp_vals < m_bin[1])

        for u_bin in unq_bins:
            unq_ix = (walks_per_unq >= u_bin[0]) & (walks_per_unq < u_bin[1])
            both_ix = m_ix * unq_ix
            print(">>>", m_bin, "/", u_bin, "=", both_ix.sum().item())
            model_perf_bins[u_bin].append(hits[both_ix].mean())
    
    fig, ax = plt.subplots()
    bar_width = 0.15
    index = np.arange(len(sp_bins)) 
    bin_width_mult = len(sp_bins) / 2 - 0.5
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
    
    plt.title(f"{dataset_name}: NBFNet", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in sp_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("Shortest Path", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=f"Walks/{metric} Percentile", frameon=False, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})

    plt.show()
    # if not os.path.isdir(os.path.join(IMG_DIR, dataset_name)):
    #     os.mkdir(os.path.join(IMG_DIR, dataset_name))
    # plt.savefig(os.path.join(IMG_DIR, dataset_name, f"SP_Control_Walks-Per-{metric}_NBFNet.png"), dpi=300, bbox_inches='tight')




def analyze_by_ppr(dataset, version):
    """
    Difference in stats between pos samples with high/low ppr
    """
    dataset_name = f"{dataset}_{version}"
    data = get_data(dataset_name, "test")

    walks_vals = data['walks']
    sp_vals = data['sp']
    tail_deg_vals = data['tail_degree']
    head_deg_vals = data['head_degree']
    ppr_vals = data['ppr']
    pprtgt_vals = data['ppr_tgt']
    unq_vals = get_unique_vals(dataset_name)

    sp_bins = [(1, 2), (2, 4), (4, 6), (6, 101)]
    ppr_bins = [(1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]

    for sbin in sp_bins:
        sp_ix = (sp_vals >= sbin[0]) & (sp_vals < sbin[1])
        for pbin in ppr_bins:
            ppr_ix = (ppr_vals >= pbin[0]) & (ppr_vals < pbin[1])
            ppr_ix = ppr_ix * sp_ix

            if ppr_ix.sum() == 0:
                continue

            print(f"\n{sbin} / {pbin}", "\n---------")
            print("  # Samples:", ppr_ix.sum())
            print("  Mean # Walks:", walks_vals[ppr_ix].mean())
            print("  Mean SP:", sp_vals[ppr_ix].mean())
            print("  Mean Head Degree:", head_deg_vals[ppr_ix].mean())
            print("  Mean Tail Degree:", tail_deg_vals[ppr_ix].mean())
            print("  Mean PPR tgt:", pprtgt_vals[ppr_ix].mean())

            # Get # unique vals by hop
            unq_vals_ix = unq_vals[ppr_ix] 
            for l in range(unq_vals_ix.shape[1]):
                print(f"  Unique l={l+1}:", unq_vals_ix[:, l].mean())



def analyze_pos_neg(dataset, version, split):
    """
    By probability
    """
    dataset = f"{dataset}_{version}"
    is_valid = "_valid" if split == "valid" else ""
    with open(os.path.join(METRIC_DIR, f'{dataset}_pos_neg_metrics{is_valid}.pkl'), 'rb') as handle:
        pos_neg_data = pickle.load(handle)

    # ppr_bins = [(1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    prob_bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]

    # for pb in prob_bins:
    #     # print(f"\n\n{pb}:\n---------")
    #     for stype in ['pos', 'neg']:
    #         ppr_vals = np.array(pos_neg_data[stype]['ppr'])
    #         prob_vals = np.array(pos_neg_data[stype]['prob'])
    #         ppr_ix = (prob_vals >= pb[0]) & (prob_vals < pb[1])

    #         ppr_vals = ppr_vals[ppr_ix]
    #         prob_vals = prob_vals[ppr_ix]
    #         sp_vals = np.array(pos_neg_data[stype]['sp'])[ppr_ix]
    #         head_deg_vals = np.array(pos_neg_data[stype]['head_deg'])[ppr_ix]
    #         tail_deg_vals = np.array(pos_neg_data[stype]['tail_deg'])[ppr_ix]
    #         walk_vals =  np.array(pos_neg_data[stype]['walks']).sum(axis=1)[ppr_ix]
    #         unq_vals =  np.array(pos_neg_data[stype]['unq'])[ppr_ix]

            # print(f"\n>>> {stype.capitalize()}:")
            # print("  # Samples:", ppr_ix.sum())
            # print("  Prob:", prob_vals.mean())
            # print("  SP:", sp_vals.mean())
            # print("  PPR:", ppr_vals.mean())
            # print("  Head Degree:", head_deg_vals.mean())
            # print("  Tail Degree:", tail_deg_vals.mean())
            # print("  Walks:", walk_vals.mean())

            # Get # unique vals by hop
            # for l in range(unq_vals.shape[1]):
            #     print(f"  Unique l={l+1}:", unq_vals[:, l].mean())


    ### Correlation for Prob
    for stype in ['pos', 'neg']:
        if stype == "pos":
            sp_vals = np.array(pos_neg_data[stype]['sp'])
            head_deg_vals = np.array(pos_neg_data[stype]['head_deg'])
            tail_deg_vals = np.array(pos_neg_data[stype]['tail_deg'])
            # walk_vals =  np.array(pos_neg_data[stype]['walks']).sum(axis=1)
            ppr_vals = np.array(pos_neg_data[stype]['ppr'])
            probs = np.array(pos_neg_data[stype]['prob'])
        else:
            sp_vals = np.array(pos_neg_data[stype]['prob']['sp'])
            head_deg_vals =np.array(pos_neg_data[stype]['prob']['head_deg'])
            tail_deg_vals = np.array(pos_neg_data[stype]['prob']['tail_deg'])
            # walk_vals =  np.array(pos_neg_data[stype]['prob']['walks']).sum(axis=1)
            ppr_vals = np.array(pos_neg_data[stype]['prob']['ppr'])    
            probs = np.array(pos_neg_data[stype]['prob']['prob'])

        sp_not100 = sp_vals != 100
        sp_vals = sp_vals[sp_not100]
        head_deg_vals = head_deg_vals[sp_not100]
        tail_deg_vals = tail_deg_vals[sp_not100]
        # walk_vals = walk_vals[sp_not100]
        ppr_vals = ppr_vals[sp_not100]
        probs = probs[sp_not100]
    

        X = []
        # for i, j, k, l, m in zip(sp_vals, head_deg_vals, tail_deg_vals, walk_vals, ppr_vals):
        #     X.append([i, j, k, l, m])
        for i, j, k, l in zip(sp_vals, head_deg_vals, tail_deg_vals, ppr_vals):
            X.append([i, j, k, l])

        X = np.array(X)
        X = StandardScaler().fit_transform(X)
        reg = LinearRegression().fit(X, probs)

        print(f"\n{stype.capitalize()} Samples:\n-------------")
        print("R =", np.sqrt(round(reg.score(X, probs), 4)))
        print(reg.coef_)



def predict_num_negs(dataset, version):
    dataset = f"{dataset}_{version}"
    with open(os.path.join(METRIC_DIR, f'{dataset}_pos_neg_metrics.pkl'), 'rb') as handle:
        pos_neg_data = pickle.load(handle)

    sp_vals = np.array(pos_neg_data["pos"]['sp'])
    head_deg_vals = np.array(pos_neg_data["pos"]['head_deg'])
    tail_deg_vals = np.array(pos_neg_data["pos"]['tail_deg'])
    # walk_vals =  np.array(pos_neg_data[stype]['walks']).sum(axis=1)
    ppr_vals = np.array(pos_neg_data["pos"]['ppr'])

    rng_50_75 = pos_neg_data['num_neg_in_prob_range'][(0.5, 0.75)]
    rng_75_100 = pos_neg_data['num_neg_in_prob_range'][(0.75, 1)]
    rng_50_100 = np.array(rng_50_75) + np.array(rng_75_100)

    X = []
    for i, j, k, l in zip(sp_vals, head_deg_vals, tail_deg_vals, ppr_vals):
        X.append([i, j, k, l])

    X = np.array(X)
    X = StandardScaler().fit_transform(X)
    reg = LinearRegression().fit(X, rng_50_100)

    print(f"\nPredict num negatives where prob>=50%:\n-------------")
    print("R =", np.sqrt(round(reg.score(X, rng_50_100), 4)))
    print(reg.coef_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    parser.add_argument("--version", type=str, default="v4")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--metric", type=str, default="ppr")
    parser.add_argument("--eps",type=float, default=1e-7)
    parser.add_argument("--alpha",type=float, default=0.15)
    args = parser.parse_args()

    # performance_by_metric(args.dataset, args.version, "walks", split="test")

    # performance_by_SP_metric(args.dataset, args.version, args.metric, args.split, eps=args.eps, alpha=args.alpha)

    # performance_by_Walks_metric(args.dataset, args.version, args.metric, args.split, eps=args.eps, alpha=args.alpha)

    # analyze_uncommon(args.dataset, args.version, split="test", eps=args.eps, alpha=args.alpha)

    # performance_by_walks_per_metric(args.dataset, args.version, "unique", args.split, eps=args.eps, alpha=args.alpha)
    # performance_by_walks_per_metric(args.dataset, args.version, "tail_degree", args.split, eps=args.eps, alpha=args.alpha)


    # metric1_vs_metric2(args.dataset, args.version, "ppr", "head_degree", eps=1e-7, alpha=0.15)
    # metric1_vs_metric2(args.dataset, args.version, "ppr", "tail_degree", eps=1e-7, alpha=0.15)

    # analyze_by_ppr(args.dataset, args.version)

    # corr_analysis(args.dataset, args.version, args.split, eps=args.eps, alpha=args.alpha)
    analyze_pos_neg(args.dataset, args.version, args.split)
    # predict_num_negs(args.dataset, args.version)


if __name__ == "__main__":
    main()
