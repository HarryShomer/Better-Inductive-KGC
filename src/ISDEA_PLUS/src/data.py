import numpy as np
import networkx as nx
import os
import torch
from torch.utils.data import DataLoader
import random
import math

def load_dict(dict_path):
    dic = {}
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[1] not in dic:
                dic[line[1]] = int(line[0])
    return dic

def load_graph(edge_path, entity_dict, relation_dict, mode):
    data = {}
    num_node = len(entity_dict)
    num_relation = len(relation_dict) * 2

    if mode == "supervision":
        edge_list = []

        G = nx.Graph()
        for j in range(num_node):
            G.add_node(j)

        with open(edge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                h = entity_dict[line[0]]
                r = relation_dict[line[1]]
                t = entity_dict[line[2]]
                G.add_edge(h, t)
                edge_list.append([h, t, r])
        
        data["G"] = G
        data["edge_list"] = edge_list

    if mode == "both":
        threshold = 0.1

        edge_list_mes = []
        edge_list_sup = []

        G_mes = nx.Graph()
        G_mes_di = nx.DiGraph()
        G_sup = nx.Graph()

        for j in range(num_node):
            G_mes.add_node(j)
            G_mes_di.add_node(j)
            G_sup.add_node(j)

        with open(edge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                h = entity_dict[line[0]]
                r = relation_dict[line[1]]
                t = entity_dict[line[2]]
                r_reverse = int(r + num_relation/2)
                if random.random() < threshold:
                    G_sup.add_edge(h, t)
                    edge_list_sup.append([h, t, r])
                    edge_list_sup.append([t, h, r_reverse])
                else:
                    G_mes.add_edge(h, t)
                    G_mes_di.add_edge(h, t)

                    edge_list_mes.append([h, t, r])
                    edge_list_mes.append([t, h, r_reverse])
        
        i = [[], [], []]
        v = []

        I = []
        V = []
        for _ in range(num_relation):
            I.append([[], [], []])
            V.append([])

        E_h = torch.zeros(num_node, num_relation)
        E_t = torch.zeros(num_node, num_relation)

        for edge in edge_list_mes:
            h = edge[0]
            t = edge[1]
            r = edge[2]

            i[0].append(h)
            i[1].append(t)
            i[2].append(r)
            v.append(float(1.0/math.sqrt(G_mes.degree(h)*G_mes.degree(t))))

            E_h[h][r] += 1
            E_t[t][r] += 1

            I[r][0].append(h)
            I[r][1].append(t)
            I[r][2].append(r)
            V[r].append(float(1.0/math.sqrt(G_mes.degree(h)*G_mes.degree(t))))

        adjs = []
        for j in range(num_relation):
            adjs.append(torch.sparse_coo_tensor(I[j], V[j], (num_node , num_node, num_relation)))
        adj = torch.sparse_coo_tensor(i, v, (num_node, num_node, num_relation))

        data["G_mes"] = G_mes
        data["G_mes_di"] = G_mes_di
        data["G_sup"] = G_sup
        data["adj"] = adj
        data["adjs"] = adjs
        data["edge_list_mes"] = edge_list_mes
        data["edge_list_sup"] = edge_list_sup
        data["E_h"] = E_h
        data["E_t"] = E_t
        data["num_node"] = num_node
        data["num_relation"] = num_relation

    if mode == "message-passing":
        edge_list = []

        G = nx.Graph()
        G_di = nx.DiGraph()
        for j in range(num_node):
            G.add_node(j)
            G_di.add_node(j)

        with open(edge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                h = entity_dict[line[0]]
                r = relation_dict[line[1]]
                t = entity_dict[line[2]]
                G.add_edge(h, t)
                G_di.add_edge(h, t)

                edge_list.append([h, t, r])
                r_reverse = int(r + num_relation/2)
                edge_list.append([t, h, r_reverse])

        i = [[], [], []]
        v = []

        I = []
        V = []
        for _ in range(num_relation):
            I.append([[], [], []])
            V.append([])

        E_h = torch.zeros(num_node, num_relation)
        E_t = torch.zeros(num_node, num_relation)

        for edge in edge_list:
            h = edge[0]
            t = edge[1]
            r = edge[2]

            i[0].append(h)
            i[1].append(t)
            i[2].append(r)
            v.append(float(1.0/math.sqrt(G.degree(h)*G.degree(t))))

            E_h[h][r] += 1
            E_t[t][r] += 1

            I[r][0].append(h)
            I[r][1].append(t)
            I[r][2].append(r)
            V[r].append(float(1.0/math.sqrt(G.degree(h)*G.degree(t))))
        
        adjs = []
        for j in range(num_relation):
            adjs.append(torch.sparse_coo_tensor(I[j], V[j], (num_node , num_node, num_relation)))
        adj = torch.sparse_coo_tensor(i, v, (num_node, num_node, num_relation))

        data["G"] = G
        data["G_di"] = G_di
        data["adj"] = adj
        data["adjs"] = adjs
        data["edge_list"] = edge_list
        data["E_h"] = E_h
        data["E_t"] = E_t
        data["num_node"] = num_node
        data["num_relation"] = num_relation
    return data


def load_data(name, data_folder, mode="train"):
    assert mode in ["train", "test"]

    data_folder = f"{data_folder}/{name}-trans" if mode == "train" else f"{data_folder}/{name}-ind"

    entities_path = os.path.join(data_folder, "entities.dict")
    relations_path = os.path.join(data_folder, "relations.dict")

    if mode == "train":
        msg_path = os.path.join(data_folder, "train.txt")
        sup_path = os.path.join(data_folder, "valid.txt")
    else:
        msg_path = os.path.join(data_folder, "observe.txt")
        sup_path = os.path.join(data_folder, "test.txt")

    entities_dict = load_dict(entities_path)
    relations_dict = load_dict(relations_path)

    if mode == "train":
        msg_data = load_graph(msg_path, entities_dict, relations_dict, "both")
    else:
        msg_data = load_graph(msg_path, entities_dict, relations_dict, "message-passing")
    
    sup_data = load_graph(sup_path, entities_dict, relations_dict, "supervision")

    return msg_data, sup_data



def load_entities_rels_new(file, ents, rels):
    """
    Read triples from file
    """    
    entity_cnt = len(ents)
    rel_cnt = len(rels)

    with open(file, "r", encoding="utf-8") as fin:
        for l in fin:
            u, r, v = l.split()
            if u not in ents: 
                ents[u] = entity_cnt
                entity_cnt += 1
            if v not in ents: 
                ents[v] = entity_cnt
                entity_cnt += 1
            if r not in rels: 
                rels[r] = rel_cnt
                rel_cnt += 1


def load_data_new(name, data_folder, mode="train", inf_graph=1):
    """
    Load graphs for new splits
    """
    assert mode in ["train", "test"]

    data_folder = os.path.join(data_folder, name)
    entities_dict, relations_dict = {}, {}

    if mode == "train":
        msg_path = os.path.join(data_folder, "train_graph.txt")
        sup_path = os.path.join(data_folder, "valid_samples.txt")

        load_entities_rels_new(msg_path, entities_dict, relations_dict)
        load_entities_rels_new(sup_path, entities_dict, relations_dict)

        msg_data = load_graph(msg_path, entities_dict, relations_dict, "both")
        sup_data = load_graph(sup_path, entities_dict, relations_dict, "supervision")
    else:
        msg_path = os.path.join(data_folder, f"test_{inf_graph-1}_graph.txt")
        sup_path = os.path.join(data_folder, f"test_{inf_graph-1}_samples.txt")

        load_entities_rels_new(msg_path, entities_dict, relations_dict)
        load_entities_rels_new(sup_path, entities_dict, relations_dict)

        msg_data = load_graph(msg_path, entities_dict, relations_dict, "message-passing")
        sup_data = load_graph(sup_path, entities_dict, relations_dict, "supervision")

    return msg_data, sup_data

