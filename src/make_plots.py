import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(FILE_DIR, "..", "figs")


def make_mean_pos_neg_sp():
    """
    For FB, and WN datasets we have an individual plot showing
    the mean Pos and Neg SP by dataset

    **Be explicit about which dataset is the transductive "parent" dataset**
    """
    # list stores [mean-pos, mean-neg]
    data = {
        "FB15k-237": {
            "Transductive": [2.26, 2.72],
            "FB v1": [3.08, 5.72],
            "FB v2": [2.43, 5.24],
            "FB v3": [2.34, 4.63],
            "FB v4": [2.24, 4.19],
            "FB-100": [2.64, 3.96], 
            "FB-75": [2.47, 3.53], 
            "FB-50": [2.54, 3.57], 
            "FB-25": [2.24, 3.37], 
        },
        "WN18RR": {
            "Transductive": [3.4, 7.53],
            "WN v1": [2.53, 9.19],
            "WN v2": [2.85, 9.89],
            "WN v3": [4.29, 9.24],
            "WN v4": [3.17, 9.73],
        },
    }

    for parent_dataset, dataset_vals in data.items():
        datasets = list(dataset_vals.keys())
        pos_vals = [x[0] for x in dataset_vals.values()]
        neg_vals = [x[1] for x in dataset_vals.values()]
        
        fig, ax = plt.subplots()

        transductive_marker_size = (len(pos_vals) + 12) ** 2
        inductive_marker_size = (len(pos_vals) + 6) ** 2

        plt.plot(np.arange(len(datasets[1:])), [pos_vals[0]] * len(datasets[1:]),  ":", color="tab:blue", 
                 label="Transductive Positive", linewidth=2)
        plt.plot(np.arange(len(datasets[1:])), [neg_vals[0]] * len(datasets[1:]),  ":", color="tab:orange",
                 label="Transductive Negative", linewidth=2)
        plt.scatter(np.arange(len(pos_vals) - 1), pos_vals[1:], marker="+", c="tab:blue", 
                    label="Inductive Positive", s=transductive_marker_size, linewidths=2)
        plt.scatter(np.arange(len(neg_vals) - 1), neg_vals[1:], marker="_", c="tab:orange", 
                    label="Inductive Negative", s=transductive_marker_size, linewidths=2)

        plt.xticks(np.arange(len(datasets[1:])), datasets[1:], fontsize=15, rotation=15) 
        plt.yticks(fontsize=15) 
        # plt.title(f"{parent_dataset} - Mean SP Distance", fontsize=14)
        # plt.xlabel("Inductive Datasets", fontsize=14)
        plt.ylabel("Mean SP Distance", fontsize=18)
        # plt.legend()
        ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.23), prop = {'size': 14})
        plt.tight_layout()

        plt.savefig(os.path.join(IMG_DIR, f"{parent_dataset}_mean_pos_neg.png"), dpi=300, bbox_inches='tight')



def mean_sp_diff_vs_ppr():
    """
    x-axis = Mean(Pos-SP) - Mean(Neg-SP)
    y-axis = PPR Hits@10
    """
    transductive_data = {
        "mean_diff": [0.46, 4.13, 0.2, 0.2, 0.46, 2.098, 0.55],
        "ppr_hits@10": [0.027, 0.462, 0.086, 0.09, 0.09, 0.302, 0.024],
    }
    e_indutive_data = {
        "mean_diff": [6.66, 7.04, 4.95, 6.56, 2.64, 2.81, 2.29, 1.95, 0.76, 0.804],
        "ppr_hits@10": [0.771, 0.744, 0.452, 0.673, 0.412, 0.476, 0.435, 0.384, 0.198, 0.225]
    }
    er_indutive_data = {
        "mean_diff": [1.319, 1.266, 1.026, 1.139, 2.582, 2.517, 2.836, 4.445],
        "ppr_hits@10": [0.222, 0.219, 0.205, 0.209, 0.158, 0.295, 0.106, 0.232] 
    }

    fig, ax = plt.subplots()

    marker_size = 12 ** 2

    plt.scatter(transductive_data['mean_diff'], transductive_data['ppr_hits@10'], marker="o", 
                c="tab:blue", label="Transductive", s=marker_size)
    plt.scatter(e_indutive_data['mean_diff'], e_indutive_data['ppr_hits@10'], marker="o", 
                c="tab:orange", label="(E) Inductive", s=marker_size)
    plt.scatter(er_indutive_data['mean_diff'], er_indutive_data['ppr_hits@10'], marker="^", 
                c="tab:green", label="(E, R) Inductive", s=marker_size)

    # Best Fit line
    all_mean_diff = transductive_data['mean_diff'] + e_indutive_data['mean_diff'] + er_indutive_data['mean_diff']
    all_ppr_hits  = transductive_data['ppr_hits@10'] + e_indutive_data['ppr_hits@10'] + er_indutive_data['ppr_hits@10']
    plt.plot(np.unique(all_mean_diff), np.poly1d(np.polyfit(all_mean_diff, all_ppr_hits, 1))(np.unique(all_mean_diff)), 
             "--", color="gray", label="Best Fit Line")
    
    # Include Pearson Corr as text box
    reg = LinearRegression().fit(np.array(all_mean_diff).reshape(-1, 1), np.array(all_ppr_hits))
    textstr=f"Pearson Corr = {round(np.sqrt(reg.score(np.array(all_mean_diff).reshape(-1, 1), np.array(all_ppr_hits))), 2)}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize=16, bbox=props)

    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    # plt.title(f"Mean SP Difference vs. PPR Hits@10", fontsize=14)
    plt.xlabel("$\Delta$ SPD", fontsize=14)
    plt.ylabel("PPR Hits@10", fontsize=14)
    # plt.legend()
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.20), prop = {'size': 12})
    plt.tight_layout()

    plt.savefig(os.path.join(IMG_DIR, f"mean_sp_diff_vs_ppr.png"), dpi=300, bbox_inches='tight')


def ppr_vs_sota_perf(kgc_task=""):
    """
    Separate plot for "(E)", "(E, R)", ""
    """
    # list stores [PPR hits@10, SOTA hits@10]
    if kgc_task == "(E)":
        data = {
            "FB v1": [41.2, 60.7],
            "FB v2": [47.6, 70.4],
            "FB v3": [43.5, 66.7],
            "FB v4": [38.4, 66.8],
            "WN v1": [77.1, 82.6],
            "WN v2": [74.4, 79.8],
            "WN v3": [45.2, 56.8],
            "WN v4": [67.3, 74.3],
            "ILPC-S": [19.8, 25.1],
            "ILPC-L": [22.5, 14.6],
        }
        plot_title = "(E) Inductive"
        offset = 4.5
        rotation = 35
        fontsize=16
        x_axis, y_axis = 0.02, 0.93
    elif kgc_task == "(E, R)":
        data = {
            "FB-25": [22.2, 27.1],
            "FB-50": [20.5, 21.8],
            "FB-75": [21.9, 32.5],
            "FB-100": [20.9, 37.1],
            "WK-25": [23.2, 30.9],
            "WK-50": [10.6, 13.5],
            "WK-75": [29.5, 36.2],
            "WK-100": [15.8, 16.9],
        }
        plot_title = "(E, R) Inductive"
        offset = 3.5
        rotation = 30
        fontsize=16
        x_axis, y_axis = 0.02, 0.93
    else:
        data = {
            "FB15k-237": [2.7, 66.6],
            "WN18RR": [46.2, 59.9],
            "CoDEx-S": [8.6, 66.3],
            "CoDEx-M": [9.0, 49.0],
            "CoDEx-L": [9.0, 47.3],
            "Hetionet": [2.4, 40.3],
            "DBPedia100k": [30.2, 41.8]
        }
        plot_title = "Transductive"
        offset = 3
        rotation = 20
        fontsize=14
        x_axis, y_axis = 0.5, 0.9

    fig, ax = plt.subplots()
    bar_width = .3
    index = np.arange(len(data)) 
    bin_width_mult = len(data) / 2  - offset
    
    ppr_perf = [x[0] for x in list(data.values())]
    sota_perf = [x[1] for x in list(data.values())]
    mean_perc_diff = np.mean([abs(x[0] - x[1]) / (0.5 * (x[0] + x[1])) for x in list(data.values())]) * 100

    ax.bar(index, np.array(ppr_perf), bar_width, label="PPR", color=CB_color_cycle[0])
    ax.bar(index+bar_width, np.array(sota_perf), bar_width, label="SOTA", color=CB_color_cycle[1])

    textstr=f"Mean % Diff = {mean_perc_diff:.0f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(x_axis, y_axis, textstr, transform=ax.transAxes, fontsize=16, bbox=props)

    # plt.title(plot_title, fontsize=15)
    plt.xticks(index + bar_width * bin_width_mult, list(data.keys()), fontsize=fontsize, rotation=rotation) 
    plt.yticks(fontsize=20)
    # plt.xlabel("Dataset", fontsize=16)
    plt.ylabel(f"Hits@10", fontsize=20)
    plt.tight_layout()
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.14), prop = {'size': 20})

    plt.savefig(os.path.join(IMG_DIR, f"{plot_title}_ppr_vs_sota.png"), dpi=300, bbox_inches='tight')


def generated_delta_spd_vs_ppr():
    """
    FB15k-237 synthetic experiment #1
    """
    sp_delta = [
        1.14, 1.31, 1.46, 1.71, 1.72,
        2.15, 1.79, 1.71, 1.71, 1.71,
        2.38, 3.89, 1.6,  1.71, 1.81,
        1.83, 1.93
    ]
    ppr_hits = [
        0.166, 0.180, 0.213, 0.309, 0.235,
        0.400, 0.319, 0.309, 0.310, 0.309,
        0.317, 0.408, 0.325, 0.309, 0.266,
        0.254, 0.259
    ]
    transductive_sp, transdutive_hits = 0.46, 2.7
    
    marker_size = len(ppr_hits) ** 2

    fig, ax = plt.subplots()
    plt.scatter([transductive_sp], [transdutive_hits], marker="^",
                c="tab:orange", label="Transductive (Original)", s=marker_size)
    plt.scatter(sp_delta, np.array(ppr_hits) * 100, marker="o", 
                c="tab:blue", label="Inductive (Generated)", s=marker_size)

    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.xlabel("$\Delta$ SPD", fontsize=20)
    plt.ylabel("PPR Hits@10", fontsize=20)
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15), prop = {'size': 14})
    plt.tight_layout()

    plt.savefig(os.path.join(IMG_DIR, f"synth_delta_spd_vs_ppr_hits.png"), dpi=300, bbox_inches='tight')


def generated_ppr_vs_size():
    """
    FB15k-237 synthetic experiment #2
    """
    ppr_hits = [
        0.166, 0.180, 0.213, 0.309, 0.235,
        0.400, 0.319, 0.309, 0.310, 0.309,
        0.317, 0.408, 0.325, 0.309, 0.266,
        0.254, 0.259
    ]
    train_size = [
        7599, 18518, 43077, 91843, 129782,
        91843, 91843, 91843, 91843, 91843,
        149635, 210109, 91843, 91843, 91843,
        91843, 91843
    ]
    test_size = [
        72688, 49057, 36054, 18370, 15179,
        4304, 13374, 18370, 21512, 18370,
        8258, 973, 11177, 18370, 36353,
        47419, 55233
    ]

    marker_size = (len(ppr_hits) + 2) ** 2

    for dtype, dsize in zip(['train', 'inference'], [train_size, test_size]):
        fig, ax = plt.subplots()
        color = "blue" if dtype == "train" else "orange"
        plt.scatter(ppr_hits, dsize, marker="*", c=f"tab:{color}", s=marker_size)

        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16) 
        plt.xlabel("PPR Hits@10", fontsize=20)
        plt.ylabel(f"# Edges in {dtype.capitalize()}", fontsize=20)
        # ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15), prop = {'size': 14})
        plt.tight_layout()

        plt.savefig(os.path.join(IMG_DIR, f"synth_ppr_hits_vs_{dtype}_size.png"), dpi=300, bbox_inches='tight')


def compare_old_perf():
    datasets = ["CoDEx-M (E)", "WN18RR (E)", "FB15k-237 (E, R)"]
    results = [[25.1, 73.8, 29.6], [6.8, 52.5, 22]]
   
    fig, ax = plt.subplots()
    bar_width = .3
    index = np.arange(len(datasets))
    bin_width_mult = 0.5

    old_bar = ax.bar(index, results[0], bar_width, label="Old Datasets")
    new_bar = ax.bar(index + bar_width, results[1], bar_width, label="New Datasets")

    plt.bar_label(old_bar, labels=[f'{x}' for x in results[0]], fontsize=16)
    plt.bar_label(new_bar, labels=[f' {x}' for x in results[1]], fontsize=16)

    # plt.title(f"Performance", fontsize=20)
    plt.xticks(index + bar_width * bin_width_mult, datasets, fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xlabel(f"Dataset", fontsize=18)
    plt.ylabel(f"Hits@10", fontsize=18)
    plt.tight_layout()
    # plt.legend(prop={'size': 16}, loc="upper left")
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15), prop = {'size': 16})

    # plt.show()
    plt.savefig(os.path.join(IMG_DIR, f"compare_old_perf.png"), dpi=300, bbox_inches='tight')


def ultra_vs_sota_perf():
    datasets = ["CoDEx-M (E)", "WN18RR (E)", "HetioNet (E)", "FB15k-237 (E, R)", "CoDEx-M (E, R)"]
    sota = [43.6, 52.5, 78.9, 30.4, 37.1]
    ultra = [46.6, 55.6, 77.7, 67.1, 68.4]

    fig, ax = plt.subplots()
    bar_width = .3
    index = np.arange(len(datasets))
    bin_width_mult = 0.5

    old_bar = ax.bar(index, sota, bar_width, label="Supervised SOTA")
    new_bar = ax.bar(index + bar_width, ultra, bar_width, label="ULTRA")

    plt.bar_label(old_bar, labels=[f'{x}' for x in sota], fontsize=11)
    plt.bar_label(new_bar, labels=[f' {x}' for x in ultra], fontsize=11)

    # plt.title(f"Performance", fontsize=20)
    plt.xticks(index + bar_width * bin_width_mult, datasets, fontsize=12.5, rotation=10)
    plt.yticks(fontsize=16)
    # plt.xlabel(f"Dataset", fontsize=18)
    plt.ylabel(f"Mean Hits@10 ", fontsize=18)
    plt.tight_layout()
    # plt.legend(prop={'size': 16}, loc="upper left")
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15), prop = {'size': 16})

    # plt.show()
    plt.savefig(os.path.join(IMG_DIR, f"ultra_vs_sota_mean.png"), dpi=300, bbox_inches='tight')


def main():
    make_mean_pos_neg_sp()
    # mean_sp_diff_vs_ppr()

    # ppr_vs_sota_perf("(E)")
    # ppr_vs_sota_perf("(E, R)")
    # ppr_vs_sota_perf("")

    # generated_delta_spd_vs_ppr()
    # generated_ppr_vs_size()

    # compare_old_perf()
    # ultra_vs_sota_perf()


    

if __name__ == "__main__":
    main()
