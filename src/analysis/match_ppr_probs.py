import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader

from util import *
from kgpy import datasets




def read_probs(args):
    """
    Samples are in same order as nbfnet probs
    """
    dataset = args.dataset
    if args.version is not None:
        dataset = "ind" + args.dataset + args.version

    is_valid = "_valid" if args.split == "valid" else ""
    samples = torch.load(os.path.join(PRED_DIR, f"nbfnet_{dataset.replace('_', '-').lower()}_samples{is_valid}.pt"))
    pos_probs = torch.load(os.path.join(PRED_DIR, f"nbfnet_{dataset.replace('_', '-').lower()}_pos_preds{is_valid}.pt"))
    neg_probs = torch.load(os.path.join(PRED_DIR, f"nbfnet_{dataset.replace('_', '-').lower()}_neg_preds{is_valid}.pt"))

    pos_probs, neg_probs = torch.sigmoid(pos_probs).squeeze(-1), torch.sigmoid(neg_probs)

    # Transductive need to be converted to our IDs
    if args.version is None:
        samples = nbf_ids_to_ours_torch(samples, dataset)

    return pos_probs.cpu(), neg_probs.cpu(), samples.cpu()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="NBFNet")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    ### Transductive or inductive
    if args.version is None:
        args.eps = 1e-6
        args.dataset_name = args.dataset  #.replace("-", "_")
    else:
        args.eps = 1e-7
        args.dataset_name = f"{args.dataset}_{args.version}"
    
    pos_probs, neg_probs, samples = read_probs(args)
    heads, tails = samples[0], samples[2]

    ### Read PPR
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    d = args.dataset_name.upper().replace("_", "-") if args.version is None else args.dataset_name
    dataset_dir = os.path.join(root_dir, "ppr", d)
    is_valid = "_val" if args.split == "valid" else ""
    ppr = torch.load(os.path.join(dataset_dir, f"sparse_adj-015_eps-{args.eps}".replace(".", "") + is_valid + ".pt"))

    ppr_ranges = [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)]
    pos_prob_sum, neg_prob_sum = {r: 0 for r in ppr_ranges}, {r: 0 for r in ppr_ranges}
    pos_prob_count, neg_prob_count = {r: 0 for r in ppr_ranges}, {r: 0 for r in ppr_ranges}

    prob_ranges = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    sampled_neg_prob_ppr = []
    sampled_neg_prob_pred = []
    sampled_neg_prob_idx = []
    sampled_neg_pos_tail = []
    num_neg_by_prob_range = {r: [] for r in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]}

    sampled_neg_preds = []
    all_pos_preds, all_pos_pprs = [], []
    sampled_corresponding_pos_pprs = []

    sampled_neg_range_ppr = []
    sampled_neg_range_pred = []
    sampled_neg_range_idx = []
    corresponding_pos_tail = []
    
    iter = 0
    BS = 10000
    for ix in tqdm(DataLoader(range(len(heads)), BS), "Merging PPR+Probs"):
        src, dst = heads[ix].long(), tails[ix].long()
        
        pos_pred = pos_probs[ix]
        all_pos_preds.extend(pos_pred.tolist())

        ppr_bs = torch.index_select(ppr, 0, src).to_dense()
        pos_ppr = ppr_bs[torch.arange(len(src)), dst]
        all_pos_pprs.extend(pos_ppr.flatten().tolist())

        neg_pred_bs = neg_probs[ix]
        # Set PPR=-1 for false negative and positive samples
        ppr_bs = torch.where(neg_pred_bs > 0, ppr_bs, -1)

        ### Sample k negatives per positive sample
        k=10
        corresponding_pos_ppr = pos_ppr.unsqueeze(-1).repeat(1, ppr_bs.size(1))

        # First filter out undesireables
        ppr_all_negs = ppr_bs[neg_pred_bs > 0]  
        pred_all_neg = neg_pred_bs[neg_pred_bs > 0]
        corresponding_pos_ppr = corresponding_pos_ppr[neg_pred_bs > 0]
        
        # Shuffle randomly and take first (# pos) * k first samples
        idx = torch.randperm(ppr_all_negs.size(0))[:len(ix) * k]
        ppr_all_negs_filt = ppr_all_negs[idx]
        pred_all_neg_filt = pred_all_neg[idx]
        corresponding_pos_ppr = corresponding_pos_ppr[idx]
        
        # sampled_neg_pprs.extend(ppr_all_negs.tolist())
        sampled_neg_preds.extend(pred_all_neg_filt.tolist())
        sampled_corresponding_pos_pprs.extend(corresponding_pos_ppr.tolist())

        for prrr in num_neg_by_prob_range:
            prrr_ix = (neg_pred_bs >= prrr[0]) & (neg_pred_bs < prrr[1])
            num_in_prr_by_pos = prrr_ix.sum(axis=1)
            num_neg_by_prob_range[prrr].extend(num_in_prr_by_pos.tolist())

        for pr in ppr_ranges:
            ### Positive
            pr_ix = (pos_ppr >= pr[0]) & (pos_ppr < pr[1])
            pos_prob_sum[pr] += pos_pred[pr_ix].sum().item()
            pos_prob_count[pr] += pr_ix.sum().item()
            num_pos_range = pr_ix.sum().item()

            ### Negative
            pr_ix = (ppr_bs >= pr[0]) & (ppr_bs < pr[1])
            neg_prob_sum[pr] += neg_pred_bs[pr_ix].sum().item()
            neg_prob_count[pr] += pr_ix.sum().item()

            ### Sample (# pos in range) * q negatives in the ppr range with indices
            q=10
            neg_pred = neg_pred_bs[pr_ix].flatten()
            neg_ppr = ppr_bs[pr_ix].flatten()
            neg_ppr_range_idx = torch.nonzero(pr_ix)

            idx = torch.randperm(pr_ix.sum().item())[:num_pos_range * q]
            neg_pred = neg_pred[idx]
            neg_ppr  = neg_ppr[idx]
            neg_ppr_range_idx = neg_ppr_range_idx[idx]

            # Fix idx, since ppr_bs is a subset of ppr, the [0] indices aren't right
            # Corresponding correct idx are in 'src' tensor
            wrong_neg_src_ix = neg_ppr_range_idx[:, 0].flatten()
            correct_neg_src_ix = torch.index_select(src, 0, wrong_neg_src_ix)
            fixed_neg_idx = torch.stack([correct_neg_src_ix, neg_ppr_range_idx[:, 1].flatten()]).t()

            # Corresponding pos tail for each negative
            correct_pos_dst_ix = torch.index_select(dst, 0, wrong_neg_src_ix)
            corresponding_pos_tail.extend(correct_pos_dst_ix.tolist())

            sampled_neg_range_ppr.extend(neg_ppr.tolist())
            sampled_neg_range_pred.extend(neg_pred.tolist())
            sampled_neg_range_idx.extend(fixed_neg_idx.tolist())


        ### Negative samples that have a probability in range
        for pr in prob_ranges:
            pr_ix = (neg_pred_bs >= pr[0]) & (neg_pred_bs < pr[1])

            ### Sample (# pos in range) * q negatives in the ppr range with indices
            q=5
            neg_pred = neg_pred_bs[pr_ix].flatten()
            neg_ppr = ppr_bs[pr_ix].flatten()
            neg_prob_range_idx = torch.nonzero(pr_ix)

            idx = torch.randperm(pr_ix.sum().item())[:len(src) * q]
            neg_pred = neg_pred[idx]
            neg_ppr  = neg_ppr[idx]
            neg_prob_range_idx = neg_prob_range_idx[idx]

            wrong_neg_src_ix = neg_prob_range_idx[:, 0].flatten()
            correct_neg_src_ix = torch.index_select(src, 0, wrong_neg_src_ix)
            fixed_neg_idx = torch.stack([correct_neg_src_ix, neg_prob_range_idx[:, 1].flatten()]).t()

            correct_pos_dst_ix = torch.index_select(dst, 0, wrong_neg_src_ix)
            sampled_neg_pos_tail.extend(correct_pos_dst_ix.tolist())

            sampled_neg_prob_ppr.extend(neg_ppr.tolist())
            sampled_neg_prob_pred.extend(neg_pred.tolist())
            sampled_neg_prob_idx.extend(fixed_neg_idx.tolist())

        iter += 1 # Keep this here!!!!


    final_probs = {"pos": defaultdict(dict), "neg": defaultdict(dict), 
                   "diff": defaultdict(dict), "num_neg_ppr_1e2": defaultdict(dict),
                   } 
    for pr in ppr_ranges:
        final_probs['pos']['probs'][pr] = pos_prob_sum[pr] / (pos_prob_count[pr] + 1e-6)
        final_probs['neg']['probs'][pr] = neg_prob_sum[pr] / (neg_prob_count[pr] + 1e-6)
        final_probs['pos']['count'][pr] = pos_prob_count[pr]
        final_probs['neg']['counts'][pr] = neg_prob_count[pr]
    

    final_probs['all_pos_probs'] = np.array(all_pos_preds)
    final_probs['all_pos_pprs'] = np.array(all_pos_pprs)
    final_probs['sampled_neg_probs'] = np.array(sampled_neg_preds)
    # final_probs['sampled_neg_pprs'] = np.array(sampled_neg_pprs)
    final_probs['corresponding_pos_ppr'] = np.array(sampled_corresponding_pos_pprs)

    final_probs['neg_ppr_samples'] = sampled_neg_range_ppr
    final_probs['neg_pred_samples'] = sampled_neg_range_pred
    final_probs['neg_idx_samples'] = sampled_neg_range_idx
    final_probs['corresponding_pos_tail'] = corresponding_pos_tail

    final_probs['neg_prob_range_ix'] = sampled_neg_prob_idx
    final_probs['neg_prob_range_prob'] = sampled_neg_prob_pred
    final_probs['neg_prob_range_ppr'] = sampled_neg_prob_ppr
    final_probs['neg_prob_range_pos_tail'] = sampled_neg_pos_tail

    final_probs['num_neg_by_prob_range'] = num_neg_by_prob_range

    is_valid = "_valid" if args.split == "valid" else ""
    with open(os.path.join(METRIC_DIR, f"{args.dataset_name}_probs_by_ppr_range{is_valid}.pkl"), "wb") as f:
        pickle.dump(final_probs, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
