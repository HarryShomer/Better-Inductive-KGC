import os
import random
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir, num_test=1):
        self.task_dir = task_dir
        self.num_test = num_test

        self.entity2id = dict()
        self.relation2id = dict()
        self.entity2id_ind = [dict() for _ in range(num_test)]

        self.tra_train = self.read_triples('train_graph.txt', self.entity2id, self.relation2id)
        self.tra_test = self.read_triples("valid_samples.txt", self.entity2id, self.relation2id)

        self.ind_train, self.ind_test = [], []
        for i in range(num_test):
            self.ind_train.append(self.read_triples(f'test_{i}_graph.txt', self.entity2id_ind[i], self.relation2id))
            self.ind_test.append(self.read_triples(f'test_{i}_samples.txt', self.entity2id_ind[i], self.relation2id))

        # Support = 25% of tra_train
        random.shuffle(self.tra_train)
        tr_thresh = int(len(self.tra_train) * 0.75)
        self.tra_train, self.tra_valid = self.tra_train[:tr_thresh], self.tra_train[tr_thresh:]

        id2relation = list(self.relation2id.keys())
        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = [len(i) for i in self.entity2id_ind]

        self.create_inv_triples()

        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')

        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for i in range(num_test):
            for filt in self.tst_filters[i]:
                self.tst_filters[i][filt] = list(self.tst_filters[i][filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)

        self.ind_KG, self.ind_sub = [], []
        for i in range(num_test):
            i1, i2 = self.load_graph(self.ind_train[i], 'inductive', i)
            self.ind_KG.append(i1)
            self.ind_sub.append(i2)
    
        self.tra_train = np.array(self.tra_valid)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.valid_q, self.valid_a = self.fix_numpy_error(self.valid_q, self.valid_a)

        self.test_q, self.test_a = [], []
        self.ind_tst_qry, self.ind_tst_ans = [], []
        for i in range(num_test):
            i1, i2 = self.load_query(self.ind_test[i])
            i1, i2 = self.fix_numpy_error(i1, i2)
            self.ind_tst_qry.append(i1) ; self.test_q.append(i1)
            self.ind_tst_ans.append(i2) ; self.test_a.append(i2)

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test  = [len(i) for i in self.test_q]

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)


    def fix_numpy_error(self, q, a):
        """
        HACK: Fix Numpy Error
        Alternative is to downgrade numpy

        Gist is that for each sample in q there may be multiple corresonding answers in a.
        So when q[i] has len(a[i]) > 1 answers, we duplicate the sample q[i] for each answer in a[i]
        """
        new_q, new_a = [], []

        for iq, ia in zip(q, a):
            if len(ia) > 1:
                for i in ia:
                    new_q.append(iq)
                    new_a.append(i)
            else:
                new_q.append(iq)
                new_a.append(ia[0])

        return new_q, new_a

    def read_triples(self, filename, ent_vocab, rel_vocab):
        triples = []
        entity_cnt, rel_cnt = len(ent_vocab), len(rel_vocab)

        with open(os.path.join(self.task_dir, filename), "r") as f:
            for line in f:
                h, r, t = line.split()

                if h not in ent_vocab:
                    ent_vocab[h] = entity_cnt
                    entity_cnt += 1
                if t not in ent_vocab:
                    ent_vocab[t] = entity_cnt
                    entity_cnt += 1
                if r not in rel_vocab:
                    rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                h, r, t = ent_vocab[h], rel_vocab[r], ent_vocab[t]

                triples.append([h,r,t])
                # triples.append([t, r+self.n_rel, h])

        return triples


    def load_graph(self, triples, mode='transductive', test_graph=0):
        n_ent = self.n_ent if mode=='transductive' else self.n_ent_ind[test_graph]
        
        KG = np.array(triples)
        idd = np.concatenate([np.expand_dims(np.arange(n_ent),1), 2*self.n_rel*np.ones((n_ent, 1)), np.expand_dims(np.arange(n_ent),1)], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:,0])), shape=(n_fact, n_ent))
        return KG, M_sub

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='transductive', test_graph=0):
        # nodes: n_node x 2 with (batch_idx, node_idx)

        if mode == 'transductive':
            KG    = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
        else:
            KG    = self.ind_KG[test_graph]
            M_sub = self.ind_sub[test_graph]
            n_ent = self.n_ent_ind[test_graph]

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train', test_graph=0):
        if data=='train':
            return self.tra_train[batch_idx]
        if data=='valid':
            query, answer = np.array(self.valid_q), np.array(self.valid_a)
            n_ent = self.n_ent
        if data=='test':
            query, answer = np.array(self.test_q[test_graph]),  np.array(self.test_a[test_graph])
            n_ent = self.n_ent_ind[test_graph]

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]

    def get_filter(self, data='valid'):
        if data == 'valid':
            filters = defaultdict(lambda: set())
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h,r)].add(t)
        else:
            filters = [defaultdict(lambda: set()) for _ in range(self.num_test)]
            for i in range(self.num_test):
                for triple in self.ind_train[i]:
                    h, r, t = triple
                    filters[i][(h,r)].add(t)
                for triple in self.ind_test[i]:
                    h, r, t = triple
                    filters[i][(h,r)].add(t)

        return filters


    def create_inv_triples(self):
        ### Add inverse triples
        inv_triples = []
        for h, r, t in self.tra_train:
            inv_triples.append([t, r+self.n_rel, h])
        self.tra_train = self.tra_train + inv_triples

        inv_triples = []
        for h, r, t in self.tra_valid:
            inv_triples.append([t, r+self.n_rel, h])
        self.tra_valid = self.tra_valid + inv_triples

        inv_triples = []
        for h, r, t in self.tra_test:
            inv_triples.append([t, r+self.n_rel, h])
        self.tra_test = self.tra_test + inv_triples

        new_triples = []
        for i in range(self.num_test):
            inv_triples = []
            for h, r, t in self.ind_train[i]:
                inv_triples.append([t, r+self.n_rel, h])
            new_triples.append(self.ind_train[i] + inv_triples)
        self.ind_train = new_triples

        new_triples = []
        for i in range(self.num_test):
            inv_triples = []
            for h, r, t in self.ind_test[i]:
                inv_triples.append([t, r+self.n_rel, h])
            new_triples.append(self.ind_test[i] + inv_triples)
        self.ind_test = new_triples
