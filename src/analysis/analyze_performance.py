import os
import copy
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_scatter import scatter
from collections import defaultdict
from torch_geometric.utils import degree
from scipy.stats import percentileofscore

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from kgpy import datasets
from util import *


def get_results(data, split):
    """
    Returns as dictionary
    """
    dataset = data.dataset_name.lower().replace("-", "_")

    df_conve = pd.read_csv(os.path.join(RESULTS_DIR, f"conve_{dataset}_trip_preds_{split}.csv"))
    df_tucker = pd.read_csv(os.path.join(RESULTS_DIR, f"tucker_{dataset}_trip_preds_{split}.csv"))
    df_nbf = pd.read_csv(os.path.join(RESULTS_DIR, f"nbfnet_{dataset}_trip_preds_{split}.csv"))
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


def get_metric(dataset, metric, split="test", eps=1e-6, alpha=0.15):
    """
    One of SP, *_degree, walks, ppr
    """
    metric = metric.lower()

    if metric == "sp":
        metric_df = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset}_SP_{split}.csv")) 
        metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])
        metric_vals = metric_df["length"].to_numpy()
    elif "degree" in metric:
        metric_df = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset}_degree_{split}.csv")) 
        metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])
        metric_vals = metric_df[metric].to_numpy()
    elif "walks" in metric:
        metric_vals = read_walks(dataset.upper().replace("-", "_"))
    elif "ppr" in metric:
        ddir = os.path.join(PPR_DIR, dataset.upper())
        metric_df = pd.read_csv(os.path.join(ddir, f"test_ppr_eps-{str(eps).replace('.', '')}_alpha-{alpha}.csv")) 
        metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])
        metric_vals = metric_df[metric].to_numpy()

    return metric_vals



######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


def metric1_vs_metric2(data, metric1, metric2, eps=1e-6, alpha=0.15):
    """
    Correlation between two metrics
    """
    dataset_name = data.dataset_name.replace("-", "_").lower()

    metric1_vals = get_metric(data.dataset_name, metric1, eps=1e-6, alpha=0.15)
    metric2_vals = get_metric(data.dataset_name, metric2, eps=1e-6, alpha=0.15)
        
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



def performance_by_metric(data, metric, split="test"):
    """
    For different ranges of the metric we get the performance of each method

    metric = ['SP', 'head_degree', 'rel_degree']
    """
    models = ['NBFNet', 'ConvE', 'TuckER']

    ### Get Raw Results
    model_results = get_results(data, split)
    dataset_name = data.dataset_name.replace("-", "_").lower()

    metric_vals = get_metric(data.dataset_name, metric)

    if metric == "SP":
        metric_name = "Shortest Path"
        if "wn" in dataset_name:
            bins = [(1, 2), (2, 4), (4, 6), (6, 9), (9, 1000)]
        else:
            bins = [(2, 3), (3, 4), (4, 1000)]
    elif "degree" in metric:
        metric_name = metric.capitalize()
        if "in_degree" in metric:
            bins = [(0, 1), (1, 3), (3, 10), (10, 25), (25, 100000)]
        else:
            bins = [(1, 3), (3, 10), (10, 25), (25, 100), (100, 1000000)]
    elif "walks" in metric:
        metric_name = "Walks Percentile"
        metric_vals = np.array([percentileofscore(metric_vals, i, kind='strict') for i in metric_vals])
        bins = [(i * 10, (i+1) * 10) for i in range(10)]

    ### Categorize results for each model by bin
    model_perf_bins = defaultdict(list)
    for m, m_results in model_results.items():
        for b in bins:
            trip_ix = (metric_vals >= b[0]) & (metric_vals < b[1])
            print(f"# Samples {b}:", sum(trip_ix))
            model_perf_bins[m].append(m_results[trip_ix].mean())
            
    fig, ax = plt.subplots()
    bar_width = 0.2
    index = np.arange(len(bins))
    bin_width_mult = len(bins) / 2 - (len(bins) - 2) * 0.5

    for ix, (model, model_data) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(model_data), bar_width, label=model, color=CB_color_cycle[ix])
        # plt.bar_label(bbb, labels=[round(x, 1) for x in model_data], fontsize=10)

    bins[-1] = (bins[-1][0], r"$\infty$")

    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in bins], fontsize=10, rotation=15) 
    plt.yticks(fontsize=18)
    plt.xlabel(metric_name, fontsize=18)
    plt.ylabel(f"Hits@10", fontsize=18)
    plt.tight_layout()
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2),  prop = {'size': 14})

    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Perf_by_{metric}.png"), dpi=300, bbox_inches='tight')



def plot_by_degree_metric(data, model, deg1, deg2, split="test"):
    """
    Control by either "{}_degree", "walks", "ppr"
    """
    deg1_vals = get_metric(data.dataset_name, deg1)
    deg2_vals = get_metric(data.dataset_name, deg2)

    ### Get Raw Results
    model_results = get_results(data, split)
    model_results = model_results[model]
    dataset_name = data.dataset_name.replace("-", "_").lower()

    if "wn" in dataset_name.lower():
        deg_bins = [(1, 3), (3, 10), (10, 1000000)]
    else:
        deg_bins = [(1, 5), (5, 25), (25, 100), (100, 1000000)]

    model_perf_bins = defaultdict(list)
    for d1_bin in deg_bins:
        d1_ix = (deg1_vals >= d1_bin[0]) & (deg1_vals < d1_bin[1])

        for d2_bin in deg_bins:
            d2_ix = (deg2_vals >= d2_bin[0]) & (deg2_vals < d2_bin[1])
            both_ix = d1_ix * d2_ix
            print(">>>", d1_bin, "/", d2_bin, "=", both_ix.sum())
            model_perf_bins[d2_bin].append(np.nan_to_num(model_results[both_ix].mean()))

    fig, ax = plt.subplots()
    bar_width = .15 if "fb" in data.dataset_name.lower() else .2
    index = np.arange(len(deg_bins)) 
    bin_width_mult = len(deg_bins) / 2  - 0.5

    deg_bins[-1] = (deg_bins[-1][0], r"$\infty$")
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])

    plt.title(f"{data.dataset_name}: {model}", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in deg_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel(deg1, fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=deg2, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})
    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Control_{deg1}_{deg2}_{model}.png"), dpi=300, bbox_inches='tight')




def plot_by_SP_metric(data, metric, model, deg_type="tail_degree", split="test", eps=1e-5):
    """
    Control by either "{}_degree", "walks", "ppr"
    """
    sp_vals = get_metric(data.dataset_name, "SP")
    metric_vals = get_metric(data.dataset_name, metric)

    ### Get Raw Results
    model_results = get_results(data, split)
    model_results = model_results[model]
    dataset_name = data.dataset_name.replace("-", "_").lower()

    if "wn" in dataset_name.lower():
        deg_bins = [(1, 5), (5, 25), (25, 100), (100,1000000)]
        sp_bins = [(1, 2), (2, 4), (4, 6), (6, 1000)]
        ppr_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    else:
        deg_bins = [(1, 3), (3, 10), (10, 25), (25, 100), (100, 1000000)]
        sp_bins = [(2, 3), (3, 4), (4, 1000)]
        ppr_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]

    walk_bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.25), (0.25, 1)]
    
    if "degree" in metric:
        metric_bins = deg_bins
        legend_title = metric_type = deg_type
    elif metric == "ppr":
        metric_bins = ppr_bins
        legend_title = metric_type = "PPR"
    else:
        metric_bins = walk_bins
        legend_title = "# of Walks"
        metric_type = "Walks"

    model_perf_bins = defaultdict(list)
    for sp_bin in sp_bins:
        sp_ix = (sp_vals >= sp_bin[0]) & (sp_vals < sp_bin[1])

        for m_bin in metric_bins:
            m_ix = (metric_vals >= m_bin[0]) & (metric_vals < m_bin[1])
            both_ix = sp_ix * m_ix
            print(">>>", sp_bin, "/", m_bin, "=", both_ix.sum())
            model_perf_bins[m_bin].append(np.nan_to_num(model_results[both_ix].mean()))

    fig, ax = plt.subplots()
    bar_width = .15 if "fb" in data.dataset_name.lower() else .2
    index = np.arange(len(sp_bins)) 
    bin_width_mult = len(sp_bins) / 2  - 0.5

    metric_bins[-1] = (metric_bins[-1][0], r"$\infty$")
    sp_bins[-1] = (sp_bins[-1][0], r"$\infty$")
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])

    plt.title(f"{data.dataset_name}: {model}", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in sp_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("Shortest Path", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=legend_title, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})
    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Control_SP_{metric_type}_{model}.png"), dpi=300, bbox_inches='tight')


def corr_analysis(data, model, split="test", eps=1e-5, alpha=0.15):
    """
    Control by either "{}_degree", "ppr"

    We make a separate plot for each Walks
    In each plot x-axis=SP and y-axis=metric
    """
    sp_vals = get_metric(data.dataset_name, "sp")
    walk_vals = get_metric(data.dataset_name, "walks")
    tail_deg_vals = get_metric(data.dataset_name, "tail_degree")
    head_deg_vals = get_metric(data.dataset_name, "head_degree")
    ppr_vals = get_metric(data.dataset_name, 'ppr')
    pprtgt_vals = get_metric(data.dataset_name, 'ppr_tgt')

    ### Get Raw Results
    model_results = get_results(data, "test")
    model_results = model_results[model]
    dataset_name = data.dataset_name.replace("-", "_").lower()
    
    sp_not100 = sp_vals != 100
    sp_vals = sp_vals[sp_not100]
    head_deg_vals = head_deg_vals[sp_not100]
    tail_deg_vals = tail_deg_vals[sp_not100]
    walk_vals = walk_vals[sp_not100]
    ppr_vals = ppr_vals[sp_not100]
    preds = model_results[sp_not100]

    ###############################
    # R^2 = Via Linear Regression
    X = []
    for i, j, k, l, m, n in zip(sp_vals, head_deg_vals, tail_deg_vals, walk_vals, ppr_vals, pprtgt_vals):
        X.append([i, j, k, l])

    X = np.array(X)
    X = StandardScaler().fit_transform(X)
    reg = LogisticRegression(penalty="none").fit(X, preds)
    print("R =", np.sqrt(round(reg.score(X, preds), 4)))
    print(reg.coef_)


def plot_by_Walks_metric(data, metric, model, deg_type="tail_degree", split="test", eps=1e-5, alpha=0.15):
    """
    Control by either "{}_degree", "ppr"

    In each plot x-axis=Walks and y-axis=metric
    """
    walks_vals = read_walks(data.dataset_name)
    metric_vals = get_metric(data.dataset_name, "ppr")

    ### Get Raw Results
    model_results = get_results(data, split)
    model_results = model_results[model]
    dataset_name = data.dataset_name.replace("-", "_").lower()

    # walk_bins = [(0, 25), (25, 50), (50, 75), (75, 100)]
    walk_bins = [(1, 3), (3, 10), (10, 100), (100, 1e7)]

    if "wn" in dataset_name.lower():
        deg_bins = [(1, 5), (5, 25), (25, 100), (100,1000000)]
        ppr_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    else:
        deg_bins = [(1, 3), (3, 10), (10, 25), (25, 100), (100, 1000000)]
        ppr_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    
    if "degree" in metric:
        metric_bins = deg_bins
        legend_title = metric_type = deg_type
    elif metric == "ppr":
        metric_bins = ppr_bins
        legend_title = metric_type = "PPR"

    # walks_vals = np.array([percentileofscore(walks_vals, i, kind='strict') for i in walks_vals])

    model_perf_bins = defaultdict(list)
    for w_bin in walk_bins:
        w_ix = (walks_vals >= w_bin[0]) & (walks_vals < w_bin[1])

        for m_bin in metric_bins:
            m_ix = (metric_vals >= m_bin[0]) & (metric_vals < m_bin[1])
            both_ix = w_ix * m_ix
            print(">>>", w_bin, "/", m_bin, "=", both_ix.sum())
            model_perf_bins[m_bin].append(np.nan_to_num(model_results[both_ix].mean()))

    fig, ax = plt.subplots()
    bar_width = .15 if "fb" in data.dataset_name.lower() else .2
    index = np.arange(len(walk_bins)) 
    bin_width_mult = len(walk_bins) / 2  - 0.5
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
    
    walk_bins[-1] = (walk_bins[-1][0], r"$\infty$")

    plt.title(f"{data.dataset_name}: {model}", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in walk_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("# of Walks", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=legend_title, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})
    plt.savefig(os.path.join(IMG_DIR, dataset_name, f"Control_Raw_Walks_{metric_type}_{model}.png"), dpi=300, bbox_inches='tight')




def performance_by_walk_ratio(data, eps, alpha, split="test"):
    """
    Ratio of shorter to longer paths
    """
    ### Get Raw Results
    model_results = get_results(data, split)['NBFNet']
    dataset_name = data.dataset_name.replace("-", "_").lower()

    sp_vals = get_metric(data.dataset_name, "sp")
    ppr_vals = get_metric(data.dataset_name, "ppr")
    walks_vals = read_walks(dataset_name.upper(), cum=False)

    # Rel to SPD
    walk_ratios = []
    for w, s in zip(walks_vals, sp_vals):
        if s == 100:
            walk_ratios.append(0)
        else:
            low_w = w[:2]
            high_w = w[2:]
            walk_ratios.append(sum(low_w) / (sum(low_w) + sum(high_w) + 1e-6))

    walk_ratios = np.nan_to_num(walk_ratios)
    walk_ratios = np.array([percentileofscore(walk_ratios, i, kind='strict') for i in walk_ratios])
    # bins = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100)]

    ppr_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    rat_bins = [(0, 10), (10, 33), (33, 66), (66, 100)]


    model_perf_bins = defaultdict(list)
    for m_bin in ppr_bins:
        m_ix = (ppr_vals >= m_bin[0]) & (ppr_vals < m_bin[1])

        for w_bin in rat_bins:
            w_ix = (walk_ratios >= w_bin[0]) & (walk_ratios < w_bin[1])
            both_ix = w_ix * m_ix
            print(">>>", w_bin, "/", m_bin, "=", both_ix.sum())
            model_perf_bins[w_bin].append(np.nan_to_num(model_results[both_ix].mean()))


    fig, ax = plt.subplots()
    bar_width = .15 if "fb" in data.dataset_name.lower() else .2
    index = np.arange(len(ppr_bins)) 
    bin_width_mult = len(ppr_bins) / 2  - 0.5
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
    

    plt.title(f"{data.dataset_name}: NBFNet", fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in ppr_bins], fontsize=12) 
    plt.yticks(fontsize=18)
    plt.xlabel("PPR", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title="Walk Ratio", frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.33), prop = {'size': 12})

    plt.show()


def analyze_uncommon(data, model, split="test", eps=1e-5, alpha=0.15):
    """
    Analyze:
        1. Low PPR samples that do well
        2. High PPR samples that do well 
    """
    walks_vals = read_walks(data.dataset_name.replace("-", "_"), cum=True)
    sp_vals = get_metric(data.dataset_name, "sp")
    tail_deg_vals = get_metric(data.dataset_name, "tail_degree")
    head_deg_vals = get_metric(data.dataset_name, "head_degree")
    ppr_vals = get_metric(data.dataset_name, "ppr")

    ### Get Raw Results
    model_results = get_results(data, split)
    model_results = model_results[model]
    dataset_name = data.dataset_name.replace("-", "_").lower()

    low_ppr_thresh = 1e-4 if "wn" in dataset_name else 5e-5
    high_ppr_thresh = 1e-2 if "wn" in dataset_name else 1e-3

    ### Low PPR analysis
    low_ppr_wrong_ix = (ppr_vals <= low_ppr_thresh) & (model_results == 0) #& (sp_vals == 1)
    low_ppr_right_ix = (ppr_vals <= low_ppr_thresh) & (model_results == 1) #& (sp_vals == 1)

    print("Low PPR Mean Stats (Wrong/Correct):\n-----------------------------")
    print("# Samples =", low_ppr_wrong_ix.sum(), low_ppr_right_ix.sum())
    print("  SP:", sp_vals[low_ppr_wrong_ix].mean(), sp_vals[low_ppr_right_ix].mean())
    print("  Walks:", walks_vals[low_ppr_wrong_ix].mean(), walks_vals[low_ppr_right_ix].mean())
    print("  Tail Degree:", tail_deg_vals[low_ppr_wrong_ix].mean(), tail_deg_vals[low_ppr_right_ix].mean())
    print("  Head Degree:", head_deg_vals[low_ppr_wrong_ix].mean(), head_deg_vals[low_ppr_right_ix].mean())

    ### High PPR analysis
    high_ppr_wrong_ix = (ppr_vals >= high_ppr_thresh) & (model_results == 0) 
    high_ppr_right_ix = (ppr_vals >= high_ppr_thresh) & (model_results == 1)

    print("\nHigh PPR Mean Stats (Wrong/Correct):\n-----------------------------")
    print("# Samples =", high_ppr_wrong_ix.sum(), high_ppr_right_ix.sum())
    print("  SP:", sp_vals[high_ppr_wrong_ix].mean(), sp_vals[high_ppr_right_ix].mean())
    print("  Walks:", walks_vals[high_ppr_wrong_ix].mean(), walks_vals[high_ppr_right_ix].mean())
    print("  Tail Degree:", tail_deg_vals[high_ppr_wrong_ix].mean(), tail_deg_vals[high_ppr_right_ix].mean())
    print("  Head Degree:", head_deg_vals[high_ppr_wrong_ix].mean(), head_deg_vals[high_ppr_right_ix].mean())


def analyze_by_ppr(data):
    """
    Difference in stats between pos samples with high/low ppr
    """
    walks_vals = read_walks(data.dataset_name.replace("-", "_"), cum=True)
    sp_vals = get_metric(data.dataset_name, "sp")
    tail_deg_vals = get_metric(data.dataset_name, "tail_degree")
    head_deg_vals = get_metric(data.dataset_name, "head_degree")
    ppr_vals = get_metric(data.dataset_name, "ppr")
    pprtgt_vals = get_metric(data.dataset_name, "ppr_tgt")
    unq_vals = get_unique_vals(data.dataset_name.replace("-", '_'))

    if "fb" in data.dataset_name.lower():
        sp_bins = [(2, 3), (3, 4), (4, 101)]
    else:
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
            print("  Mean PPR Tgt:", pprtgt_vals[ppr_ix].mean())

            # Get # unique vals by hop
            unq_vals_ix = unq_vals[ppr_ix] 
            for l in range(unq_vals_ix.shape[1]):
                print(f"  Unique l={l+1}:", unq_vals_ix[:, l].mean())


def analyze_pos_neg(dataset):
    """
    By probability
    """
    with open(os.path.join(METRIC_DIR, f'{dataset}_pos_neg_metrics.pkl'), 'rb') as handle:
        pos_neg_data = pickle.load(handle)

    # ppr_bins = [(1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    # prob_bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]

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
            head_deg_vals =np.array(pos_neg_data[stype]['head_deg'])
            tail_deg_vals = np.array(pos_neg_data[stype]['tail_deg'])
            # walk_vals =  np.array(pos_neg_data[stype]['walks']).sum(axis=1)
            ppr_vals = np.array(pos_neg_data[stype]['ppr'])
            head_rels = np.array(pos_neg_data[stype]['rel_deg'])
            probs = np.array(pos_neg_data[stype]['prob'])
        else:
            sp_vals = np.array(pos_neg_data[stype]['prob']['sp'])
            head_deg_vals =np.array(pos_neg_data[stype]['prob']['head_deg'])
            tail_deg_vals = np.array(pos_neg_data[stype]['prob']['tail_deg'])
            # walk_vals =  np.array(pos_neg_data[stype]['prob']['walks']).sum(axis=1)
            ppr_vals = np.array(pos_neg_data[stype]['prob']['ppr'])    
            head_rels = np.array(pos_neg_data[stype]['prob']['rel_deg'])
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
        for i, j, k, l, m in zip(sp_vals, head_deg_vals, tail_deg_vals, head_rels, ppr_vals):
            X.append([i, j, k, l])

        X = np.array(X)
        X = StandardScaler().fit_transform(X)
        reg = LinearRegression().fit(X, probs)

        print(f"\n{stype.capitalize()} Samples:\n-------------")
        print("R =", np.sqrt(round(reg.score(X, probs), 4)))
        print(reg.coef_)


def predict_num_negs(dataset):
    with open(os.path.join(METRIC_DIR, f'{dataset}_pos_neg_metrics.pkl'), 'rb') as handle:
        pos_neg_data = pickle.load(handle)

    sp_vals = np.array(pos_neg_data["pos"]['sp'])
    head_deg_vals = np.array(pos_neg_data["pos"]['head_deg'])
    tail_deg_vals = np.array(pos_neg_data["pos"]['tail_deg'])
    # walk_vals =  np.array(pos_neg_data[stype]['walks']).sum(axis=1)
    ppr_vals = np.array(pos_neg_data["pos"]['ppr'])
    head_rels = np.array(pos_neg_data["pos"]['rel_deg'])

    rng_50_75 = pos_neg_data['num_neg_in_prob_range'][(0.5, 0.75)]
    rng_75_100 = pos_neg_data['num_neg_in_prob_range'][(0.75, 1)]
    rng_50_100 = np.array(rng_50_75) + np.array(rng_75_100)

    X = []
    for i, j, k, l, m in zip(sp_vals, head_deg_vals, tail_deg_vals, head_rels, ppr_vals):
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
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eps",type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.15)
    args = parser.parse_args()

    data = getattr(datasets, args.dataset.upper().replace("-", "_"))(inverse=True)

    # performance_by_walk_ratio(data, args.eps, args.alpha, split=args.split)

    # performance_by_metric(data, "walks")

    # plot_by_SP_metric(data, "ppr", "NBFNet", eps=args.eps)
    # plot_by_SP_metric(data, "ppr", "TuckER", eps=args.eps)
    # plot_by_SP_metric(data, "ppr", "AnyBURL", eps=args.eps)

    # plot_by_Walks_metric(data, "ppr", "NBFNet", eps=args.eps, alpha=args.alpha)

    # analyze_uncommon(data, "NBFNet", eps=args.eps, alpha=args.alpha)
    # plot_by_degree_metric(data, "NBFNet", "tail_degree", "head_degree")

    # metric1_vs_metric2(data, "ppr", "head_degree")
    # metric1_vs_metric2(data, "ppr", "tail_degree")

    # analyze_by_ppr(data)

    # corr_analysis(data, "NBFNet", eps=args.eps, alpha=args.alpha)
    # analyze_pos_neg(args.dataset)
    predict_num_negs(args.dataset)


if __name__ == "__main__":
    main()
