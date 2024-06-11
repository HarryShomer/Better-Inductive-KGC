import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(FILE_DIR, "..", "..", "imgs")
METRIC_DIR = os.path.join(FILE_DIR, "..", "..", "data", "metrics", "ogb")
RESULTS_DIR = os.path.join(FILE_DIR, "..", "..", "data", "model_results", "ogb")

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def get_results(dataset):
    """
    Returns as dictionary
    """
    gcn_preds = torch.load(os.path.join(RESULTS_DIR, f"{dataset}_GCN_test_preds_seed-1.pt"))
    buddy_preds = torch.load(os.path.join(RESULTS_DIR,f"{dataset}_BUDDY_test_preds_seed-1.pt"))
    ncnc_preds = torch.load(os.path.join(RESULTS_DIR,f"{dataset}_NCNC_test_preds_seed-1.pt"))
    lpformer_preds = torch.load(os.path.join(RESULTS_DIR,f"{dataset}_lpformer_test_preds_seed-1.pt"))

    df_gcn = pd.DataFrame(gcn_preds.tolist(), columns=['src', 'dst', 'pred'])
    df_buddy = pd.DataFrame(buddy_preds.tolist(), columns=['src', 'dst', 'pred'])
    df_ncnc = pd.DataFrame(ncnc_preds.tolist(), columns=['src', 'dst', 'pred'])
    df_lpformer = pd.DataFrame(lpformer_preds.tolist(), columns=['src', 'dst', 'pred'])
    
    df_gcn = df_gcn.sort_values(by=['src', 'dst'])
    df_buddy = df_buddy.sort_values(by=['src', 'dst'])
    df_ncnc = df_ncnc.sort_values(by=['src', 'dst'])
    df_lpformer = df_lpformer.sort_values(by=['src', 'dst'])

    models = {
        "GCN": df_gcn['pred'].to_numpy(),
        "BUDDY": df_buddy['pred'].to_numpy(),
        "NCNC": df_ncnc['pred'].to_numpy(),
        "LPFormer": df_lpformer['pred'].to_numpy()
    }

    return models 



def plot_SP_by_metric(dataset, metric, model):
    """
    metric = one of ['CN', 'PPR', 'min_degree', 'max_degree', 'mean_degree'] 
    """
    model_results = get_results(dataset)
    model_results = model_results[model]

    df_metric = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset}_metrics.csv"), index_col=0)
    df_metric = df_metric.sort_values(by=['src', 'dst'])

    sp_vals = df_metric['SP']
    metric_vals = df_metric[metric]

    if "collab" in dataset:
        sp_bins = [(2, 3), (3, 4), (4, 6), (6, 1000)]
    else:
        sp_bins = [(2, 3), (3, 1000)]

    if metric == "CN":
        sp_bins = [(1, 3), (3, 6), (6, 1000)]
        metric_bins = [(0, 1), (1, 3), (3, 10), (10, 100000)]
    elif metric == "PPR":
        if "collab" in dataset:
            metric_bins = [(0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1)]
        else:
            metric_bins = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1)]
    elif "degree" in metric: 
        metric_bins = [(1, 10), (10, 25), (25, 50), (50, 100),  (100, 1000000)]

    model_perf_bins = defaultdict(list)
    for sp_bin in sp_bins:
        sp_ix = (sp_vals >= sp_bin[0]) & (sp_vals < sp_bin[1])

        for m_bin in metric_bins:
            deg_ix = (metric_vals >= m_bin[0]) & (metric_vals < m_bin[1])
            both_ix = sp_ix * deg_ix
            print(">>>", sp_bin, "/", m_bin, "=", both_ix.sum())
            model_perf_bins[m_bin].append(model_results[both_ix].mean())

    fig, ax = plt.subplots()
    bar_width = .15
    index = np.arange(len(sp_bins)) 
    bin_width_mult = len(m_bin) / 2 + 0.5

    metric_bins[-1] = (metric_bins[-1][0], r"$\infty$")
    sp_bins[-1] = (sp_bins[-1][0], r"$\infty$")
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])

    plt.title(f"{model} - {dataset}", fontsize=18)
    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in sp_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("Distance", fontsize=16)
    plt.ylabel(f"Performance", fontsize=16)
    plt.tight_layout()
    ax.legend(title=metric, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.333), prop = {'size': 12})

    if not os.path.isdir(os.path.join(IMG_DIR, dataset)):
        os.mkdir(os.path.join(IMG_DIR, dataset))
    plt.savefig(os.path.join(IMG_DIR, dataset, f"{model}_Perf_by_SP-{metric}.png"), dpi=300, bbox_inches='tight')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbl-collab")
    parser.add_argument("--metric", type=str, default="CN")
    args = parser.parse_args()

    for model in ['GCN', 'BUDDY', 'NCNC', 'LPFormer']:
        plot_SP_by_metric(args.dataset, args.metric, model)
    

    


if __name__ == "__main__":
    main()
