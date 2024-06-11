import os
import torch
import pickle 
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url
import kgpy

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(FILE_DIR, "..", "..", "imgs")
PPR_DIR = os.path.join(FILE_DIR, "..", "..", "data", "ppr")
RULES_DIR = os.path.join(FILE_DIR, "..", "..", "data", "rules")
METRIC_DIR = os.path.join(FILE_DIR, "..", "..", "data", "metrics")
RESULTS_DIR = os.path.join(FILE_DIR, "..", "..", "data", "model_results")
PRED_DIR = os.path.join(FILE_DIR, "..", "..", "data", "preds")


def get_unique_vals(dataset_name):
    """
    Unique from l=1 to l=L
    """
    unq_df = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset_name}_Unique.csv")) 
    unq_df = unq_df.sort_values(by=['head', 'rel', 'tail'])
    unq_vals = []

    for ix, unqs in enumerate(unq_df.to_dict("records")):
        aaa = []
        for ix in range(1, 7):
            aaa.append(unqs.get(f'unique_{ix}', 0))
        unq_vals.append(aaa)        
    
    return np.array(unq_vals)


def get_metric_for_spd(sp_vals, m_vals):
    """
    metric when length=spd
    """
    metric_vals = []
    for ix, w in enumerate(m_vals):
        isp = int(sp_vals[ix])
        if isp <= 6:
            metric_vals.append(w[isp-1])
        else:
            metric_vals.append(0)

    return np.array(metric_vals)


def read_walks(dataset_name, cum=True):
    """
    Read walks data

    When cum=True, get cumulative #
    """
    walks_df = pd.read_csv(os.path.join(METRIC_DIR, f"{dataset_name}_Walks.csv")) 
    walks_df = walks_df.sort_values(by=['head', 'rel', 'tail'])
    walks_vals = []

    for ix, walks in enumerate(walks_df.to_dict("records")):
        aaa = []

        for ix in range(1, 7):
            aaa.append(walks.get(f'length_{ix}', 0))

        if cum:
            walks_vals.append(sum(aaa))
        else:
            walks_vals.append(aaa)        
    
    return np.array(walks_vals)



def read_anyburl_results(data):
    """
    1. Convert to IDs
    2. Head/Test results are on same line
    3. Results are if explanation exist. Wrong when None
    """
    dataset = data.dataset_name.upper()
    exp_file = os.path.join(RESULTS_DIR, "..", "explanations", f"{dataset}_AnyBURL_Test.pkl")
    with open(exp_file, 'rb') as handle:
        results = pickle.load(handle)

    ent2idx = data.entity2idx
    rel2idx = data.relation2idx
    num_rels = data.num_non_inv_rels

    all_results = []
    for k in results:
        h, r, t = ent2idx[k[0]], rel2idx[k[1]], ent2idx[k[2]]

        head_result = float(k[3] is not None)
        tail_result = float(k[4] is not None)

        all_results.append([h, r, t, tail_result])
        all_results.append([t, r+num_rels, h, head_result]) # Concat inv rule

    df = pd.DataFrame(all_results, columns=["head", "rel", "tail", "hits@10"])
    df = df.sort_values(by=['head', 'rel', 'tail'])

    return df['hits@10'].to_numpy()



def get_trip_ppr(data, eps):
    """
    Get PPR for all samples

    NBFNet only inits head, so only calc (h, t) for fact (h, r, t)
    """
    dataset = data.dataset_name.upper()
    ppr_file = os.path.join(PPR_DIR, dataset, f"test_ppr_eps-{str(eps).replace('.', '')}.pkl")
    with open(ppr_file, 'rb') as handle:
        ppr_vals = pickle.load(handle)

    return ppr_vals


def nbf_ids_to_ours_torch(samples, dataset_name):
    """
    Only for transductive setting
    """
    # Map nbf_id -> our id
    nbf_ents, nbf_rels = {}, {}

    data = getattr(kgpy.datasets, dataset_name.upper().replace("-", "_"))(inverse=True)
    dataset_name = dataset_name.replace("-", "").replace("_", "")

    with open(os.path.join(FILE_DIR, "..", "..", "data", "nbf_ids", f"{dataset_name.lower()}_ent_ids.txt"), "r") as f:
        for ent in f:
            ent_cs = ent.split()  # ID, Name
            nbf_ents[int(ent_cs[0].strip())] = data.entity2idx[ent_cs[1].strip()]

    with open(os.path.join(FILE_DIR, "..", "..", "data", "nbf_ids", f"{dataset_name.lower()}_rel_ids.txt"), "r") as f:
        for rel in f:
            rel_cs = rel.split()  # ID, Name
            nbf_rels[int(rel_cs[0].strip())] = data.relation2idx[rel_cs[1].strip()]
            nbf_rels[int(rel_cs[0].strip()) + data.num_non_inv_rels] = data.relation2idx[rel_cs[1].strip()] + data.num_non_inv_rels
    
    new_samples = []
    for t in samples.t().tolist():
        new_samples.append((nbf_ents[t[0]], nbf_rels[t[1]], nbf_ents[t[2]]))

    return torch.Tensor(new_samples).t()



def nbf_ids_to_ours(data, df):
    """
    Conversions are found in data/nbf_ids files
    """
    # Map nbf_id -> our id
    nbf_ents, nbf_rels = {}, {}
    dataset = data.dataset_name.lower().replace("-", "")
    num_rels = data.num_relations // 2

    with open(os.path.join(FILE_DIR, "..", "..", "data", "nbf_ids", f"{dataset}_ent_ids.txt"), "r") as f:
        for ent in f:
            ent_cs = ent.split()  # ID, Name
            nbf_ents[int(ent_cs[0].strip())] = data.entity2idx[ent_cs[1].strip()]

    with open(os.path.join(FILE_DIR, "..", "..", "data", "nbf_ids", f"{dataset}_rel_ids.txt"), "r") as f:
        for rel in f:
            rel_cs = rel.split()  # ID, Name
            nbf_rels[int(rel_cs[0].strip())] = data.relation2idx[rel_cs[1].strip()]
            nbf_rels[int(rel_cs[0].strip()) + num_rels] = data.relation2idx[rel_cs[1].strip()] + num_rels
    
    df['head'] = df.apply(lambda x: nbf_ents[int(x['head'])] , axis=1)
    df['tail'] = df.apply(lambda x: nbf_ents[int(x['tail'])] , axis=1)
    df['rel'] = df.apply(lambda x: nbf_rels[int(x['rel'])] , axis=1)

    return df


def our_ids_to_nbf(data, df):
    """
    Conversions are found in data/nbf_ids files
    """
    # Map nbf_id -> our id
    nbf_ents, nbf_rels = {}, {}
    dataset = data.dataset_name.lower().replace("-", "")
    num_rels = data.num_relations // 2

    with open(os.path.join(FILE_DIR, "..", "..", "data", "nbf_ids", f"{dataset}_ent_ids.txt"), "r") as f:
        for ent in f:
            ent_cs = ent.split()  # ID, Name
            nbf_ents[data.entity2idx[ent_cs[1].strip()]] = int(ent_cs[0].strip())

    with open(os.path.join(FILE_DIR, "..", "..", "data", "nbf_ids", f"{dataset}_rel_ids.txt"), "r") as f:
        for rel in f:
            rel_cs = rel.split()  # ID, Name
            nbf_rels[data.relation2idx[rel_cs[1].strip()]] = int(rel_cs[0].strip())
            nbf_rels[data.relation2idx[rel_cs[1].strip()] + num_rels] = int(rel_cs[0].strip()) + num_rels

    

    df['head'] = df.apply(lambda x: nbf_ents[int(x['head'])] , axis=1)
    df['tail'] = df.apply(lambda x: nbf_ents[int(x['tail'])] , axis=1)
    df['rel'] = df.apply(lambda x: nbf_rels[int(x['rel'])] , axis=1)

    return df


def read_rule_nums(data, weighted=False):
    """
    (h, r, t, # rules)
    """
    dataset = data.dataset_name.lower().replace("-", "_")
    
    suffix = "_weight" if weighted else ""
    with open(os.path.join(RULES_DIR, f'{dataset}_rules_per_trip{suffix}.pkl'), 'rb') as handle:
        all_rules = pickle.load(handle)

    ent2idx = data.entity2idx
    rel2idx = data.relation2idx

    all_rules_id = []
    for r in all_rules:
        all_rules_id.append([ent2idx[r[0]], rel2idx[r[1]], ent2idx[r[2]], r[-1]])

        # Also append reciprocal
        inv_rel = rel2idx[r[1]] + data.num_non_inv_rels
        all_rules_id.append([ent2idx[r[2]], inv_rel, ent2idx[r[0]], r[-1]])

    return pd.DataFrame(all_rules_id, columns=['head', 'rel', 'tail', 'num'])


def read_rules(data):
    """
    Read rules

    In process, convert from raw names to IDs we use
    """
    dataset = data.dataset_name.lower().replace("-", "_")

    with open(os.path.join(RULES_DIR, f'{dataset}_cyclic_rules.pkl'), 'rb') as handle:
        cyclic_rules = pickle.load(handle)
    with open(os.path.join(RULES_DIR, f'{dataset}_acyclic_rules.pkl'), 'rb') as handle:
        acyclic_rules = pickle.load(handle)

    ent2idx = data.entity2idx
    rel2idx = data.relation2idx

    id_cyclic_rules = {}
    for k, v in cyclic_rules.items():
        id_cyclic_rules[rel2idx[k]] = v

    id_acyclic_rules = {}
    # for k, v in acyclic_rules.items():
    #     k = (rel2idx[k[0]], ent2idx[k[1]])
    #     id_acyclic_rules[k] = v

    return id_cyclic_rules, id_acyclic_rules


class IndRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ]
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def dataset_name(self):
        return f"{self.name}_{self.version}"

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name
