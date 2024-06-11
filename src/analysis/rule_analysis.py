import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression


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

    models = {
        "NBFNet": df_nbf['hits@10'].to_numpy(),
        "ConvE": df_conve['hits@10'].to_numpy(),
        "TuckER": df_tucker['hits@10'].to_numpy()
    }

    return models 




def plot_deg_vs_rules(data, weighted=False, split="test"):
    """
    """
    dataset = data.dataset_name.lower().replace("-", "_")

    metric_df = pd.read_csv(os.path.join(METRIC_DIR, f"{data.dataset_name}_degree_{split}.csv")) 
    metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])
    deg_vals = metric_df["tail_degree"].to_numpy()
    
    rules_df = read_rule_nums(data, weighted)
    rules_df = rules_df.sort_values(by=['head', 'rel', 'tail'])
    num_rules = rules_df['num'].to_numpy()


    reg = LinearRegression().fit(deg_vals.reshape(-1, 1), num_rules)
    textstr=f"R^2 = {round(reg.score(deg_vals.reshape(-1, 1), num_rules), 3)}"
    print(textstr)


    fig, ax = plt.subplots()
    plt.scatter(deg_vals, num_rules)
    plt.plot(np.unique(deg_vals), np.poly1d(np.polyfit(deg_vals, num_rules, 1))(np.unique(deg_vals)), color="red")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=18) 
    plt.title(f"{data.dataset_name}: Tail Degree vs. # of Potential Rules", fontsize=14)
    plt.xlabel("Train Degree", fontsize=18)
    plt.ylabel("# Possible Rules", fontsize=18)

    # these are matplotlib.patch.Patch properties
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.75, textstr, transform=ax.transAxes, fontsize=14, bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, dataset, "rules", f"train_degree_vs_num_rules.png"), dpi=300, bbox_inches='tight')



def plot_perf_by_rules(data, weighted=False, model="NBFNet", split="test"):
    """
    By # of rules controlling for SP
    """
    dataset = data.dataset_name.lower().replace("-", "_")

    if "wn" in dataset:
        sp_bins = [(1, 2), (2, 4), (4, 6), (6, 9), (9, 1000)]
        # rule_bins = [(0, 10), (10, 25), (25, 50), (50, 10000000)]
        rule_bins = [(0, 1), (1, 1000), (1000, 2500), (2500, 10000), (10000, 10e9)]
    else:
        sp_bins = [(2, 3), (3, 4), (4, 1000)]
        rule_bins = [(0, 1), (1, 100), (100, 1000), (1000, 2500), (2500, 10e9)]
    

    metric_df = pd.read_csv(os.path.join(METRIC_DIR, f"{data.dataset_name}_SP_{split}.csv")) 
    metric_df = metric_df.sort_values(by=['head', 'rel', 'tail'])
    sp_vals = metric_df["length"].to_numpy()

    model_results = get_results(data, split)
    model_results = model_results[model]

    rules_df = read_rule_nums(data, weighted)
    rules_df = rules_df.sort_values(by=['head', 'rel', 'tail'])
    num_rules = rules_df['num'].to_numpy()

    # num_rules = num_rules / num_rules.max()

    model_perf_bins = defaultdict(list)
    for sp_bin in sp_bins:
        sp_ix = (sp_vals >= sp_bin[0]) & (sp_vals < sp_bin[1])

        for r_bin in rule_bins:
            r_ix = (num_rules >= r_bin[0]) & (num_rules < r_bin[1])
            both_ix = sp_ix * r_ix
            print(">>>", sp_bin, "/", r_bin, "=", both_ix.sum())
            model_perf_bins[r_bin].append(model_results[both_ix].mean())


    fig, ax = plt.subplots()
    bar_width = .15
    index = np.arange(len(sp_bins)) 
    bin_width_mult = len(rule_bins) / 2 - 0.5

    rule_bins[-1] = (rule_bins[-1][0], r"$\infty$")
    sp_bins[-1] = (sp_bins[-1][0], r"$\infty$")
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
        

    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in sp_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("Shortest Path", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=f"# Rules", frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.26), prop = {'size': 12})

    plt.savefig(os.path.join(IMG_DIR, dataset, "rules", f"{model}_Perf_By_SP_Num-Rules.png"), dpi=300, bbox_inches='tight')
    

#########################################################
#########################################################
#########################################################

def read_trip_rule_data(data):
    """
    Given: (h, r, t, "ent predicting", rule_id, rule type, rule length)

    Convert h, r, t to our IDs

    NOTE: Later when training will include all inverse edges to always predict the tail
          Therefore we can just use the tail to group the triples
    """
    data_dir = os.path.join(FILE_DIR, "..", "..", "data", "explanations")
    
    dataset = data.dataset_name.upper().replace("-", "_")
    with open(os.path.join(data_dir, f"{dataset}_AnyBURL.pkl"),  'rb') as handle:
        trip2exps = pickle.load(handle)

    ent2idx = data.entity2idx
    rel2idx = data.relation2idx


    all_trips = []
    for tr in trip2exps:
        if tr[3] == "tail":  
            h, r, t = ent2idx[tr[0]], rel2idx[tr[1]], ent2idx[tr[2]]
            all_trips.append([h, r, t, tr[5], tr[6]])

            # Add inverse
            all_trips.append([t, r + data.num_non_inv_rels, h, tr[5], tr[6]])
    
    return pd.DataFrame(all_trips, columns=['head', 'rel', 'tail', 'rule_type', 'rule_length'])


def plot_control_sp_by_rules(data, split="test"):
    """
    """
    dataset = data.dataset_name.lower().replace("-", "_")

    sp_bins = [(1, 2), (2, 4), (4, 6), (6, 1000)]
    rule_bins = [(1, 2), (2, 3), (3, 4), (4, 1000)]

    metric_df = pd.read_csv(os.path.join(RESULTS_DIR, f"nbfnet_{data.dataset_name.lower()}-Control_SP_preds_{split}.csv")) 
    rule_df = read_trip_rule_data(data)    
    merged_df = pd.merge(metric_df, rule_df, how="left", on=['head', 'rel', 'tail'])

    sp_vals = merged_df["sp"].to_numpy()
    hits = merged_df["hits@10"].to_numpy()
    rule_length = merged_df['rule_length'].to_numpy()

    model_perf_bins = defaultdict(list)
    for sp_bin in sp_bins:
        sp_ix = (sp_vals >= sp_bin[0]) & (sp_vals < sp_bin[1])

        for r_bin in rule_bins:
            r_ix = (rule_length >= r_bin[0]) & (rule_length < r_bin[1])
            both_ix = sp_ix * r_ix
            print(">>>", sp_bin, "/", r_bin, "=", both_ix.sum())
            model_perf_bins[r_bin].append(hits[both_ix].mean())


    fig, ax = plt.subplots()
    bar_width = .15
    index = np.arange(len(sp_bins)) 
    bin_width_mult = len(rule_bins) / 2 - 0.5

    rule_bins[-1] = (rule_bins[-1][0], r"$\infty$")
    sp_bins[-1] = (sp_bins[-1][0], r"$\infty$")
    
    for ix, (binname, bindata) in enumerate(model_perf_bins.items()):
        bbb = ax.bar(index + bar_width * ix, np.array(bindata), bar_width, label=f"{binname}", color=CB_color_cycle[ix])
        

    plt.xticks(index + bar_width * bin_width_mult, [f"[{b[0]}, {b[1]})" for b in sp_bins], fontsize=14) 
    plt.yticks(fontsize=18)
    plt.xlabel("Shortest Path", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=16)
    plt.tight_layout()
    ax.legend(title=f"Rule Length", frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.26), prop = {'size': 12})

    plt.savefig(os.path.join(IMG_DIR, f"NBF-Control_Perf_By_SP_Rule-Length.png"), dpi=300, bbox_inches='tight')
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--weight", help="Weight rules by conf", action='store_true', default=False)
    args = parser.parse_args()

    data = getattr(datasets, args.dataset.upper().replace("-", "_"))(inverse=True)

    # plot_perf_by_rules(data, weighted=args.weight)
    # plot_deg_vs_rules(data, weighted=args.weight)

    plot_control_sp_by_rules(data, split=args.split)


if __name__ == "__main__":
    main()
