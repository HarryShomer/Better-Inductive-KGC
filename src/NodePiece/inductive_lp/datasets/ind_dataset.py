import os
import numpy as np
from pykeen.datasets import PathDataSet, TriplesFactory, DataSet


####################
# NEW Datasets
####################


def load_file(triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):
    triplets = []
    entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

    with open(triplet_file, "r", encoding="utf-8") as fin:
        for l in fin:
            u, r, v = l.split()
            if u not in inv_entity_vocab:
                inv_entity_vocab[u] = entity_cnt
                entity_cnt += 1
            if v not in inv_entity_vocab:
                inv_entity_vocab[v] = entity_cnt
                entity_cnt += 1
            if r not in inv_rel_vocab:
                inv_rel_vocab[r] = rel_cnt
                rel_cnt += 1
            # u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

            triplets.append((u, r, v))

    return triplets, inv_entity_vocab, inv_rel_vocab



# class New_Ind_E(DataSet):
#     def __init__(self,
#                  dataset: str,
#                  path: str,
#                  create_inverse_triples: bool = True
#                  ):
        
#         tr_ent_vocab, te_ent_vocab, rel_vocab = {}, {}, {}
#         train_trips, tr_ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, "train_graph.txt"), 
#                                                          tr_ent_vocab, rel_vocab)
#         valid_trips, tr_ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, "valid.txt"), 
#                                                          tr_ent_vocab, rel_vocab)
#         inf_trips, te_ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, "inf_graph.txt"), 
#                                                          te_ent_vocab, rel_vocab)
#         test_trips, te_ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, "test.txt"), 
#                                                          te_ent_vocab, rel_vocab)
        
#         # Add inverse relations to mapping
#         num_rels = len(rel_vocab)
#         non_inv_rels = list(rel_vocab.keys())
#         for r in non_inv_rels:
#             rel_vocab[r + "_inverse"] = num_rels
#             num_rels += 1

#         self.transductive_part = TriplesFactory(
#             triples=np.array(train_trips, dtype=str),
#             entity_to_id=tr_ent_vocab,
#             relation_to_id=rel_vocab,
#             create_inverse_triples=create_inverse_triples
#         )

#         # Really val on train graph (even under old setting...)
#         # But we'll keep naming convention for convenience
#         self.inductive_val = TriplesFactory(
#             triples=np.array(valid_trips, dtype=str),
#             entity_to_id=tr_ent_vocab,
#             relation_to_id=rel_vocab,
#         )

#         self.inductive_inference = TriplesFactory(
#             triples=np.array(inf_trips, dtype=str),
#             entity_to_id=te_ent_vocab,
#             relation_to_id=rel_vocab,
#             create_inverse_triples=create_inverse_triples
#         )

#         self.inductive_test = TriplesFactory(
#             triples=np.array(test_trips, dtype=str),
#             entity_to_id=te_ent_vocab,
#             relation_to_id=rel_vocab,
#         )

#         # Trust me
#         self.transductive_part._num_relations = len(rel_vocab)
#         self.inductive_val._num_relations = len(rel_vocab)
#         self.inductive_inference._num_relations = len(rel_vocab)
#         self.inductive_test._num_relations = len(rel_vocab)


#         # print("Read Data!")
#         # exit()


class New_Ind(DataSet):
    def __init__(self,
                 dataset: str,
                 path: str,
                 num_test: int = 2,
                 create_inverse_triples: bool = True
                 ):
        ent_vocab, rel_vocab = {}, {}
        train_trips, ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, "train_graph.txt"), 
                                                         ent_vocab, rel_vocab)
        valid_trips, ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, "valid_samples.txt"), 
                                                         ent_vocab, rel_vocab)
        
        # Needs to be done b4 mapping so we know the number of rels to create inverse
        inf_ent_vocabs = []
        all_inf_trips, all_test_trips = [], []
        for i in range(num_test):
            te_ent_vocab = {}
            inf_trips, te_ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, f"test_{i}_graph.txt"), 
                                                            te_ent_vocab, rel_vocab)
            test_trips, te_ent_vocab, rel_vocab = load_file(os.path.join(path, dataset, f"test_{i}_samples.txt"), 
                                                            te_ent_vocab, rel_vocab)
            all_inf_trips.append(inf_trips)
            all_test_trips.append(test_trips)
            inf_ent_vocabs.append(te_ent_vocab)


        # Add inverse relations to mapping
        num_rels = len(rel_vocab)
        non_inv_rels = list(rel_vocab.keys())
        for r in non_inv_rels:
            rel_vocab[r + "_inverse"] = num_rels
            num_rels += 1


        self.transductive_part = TriplesFactory(
            triples=np.array(train_trips, dtype=str),
            entity_to_id=ent_vocab,
            relation_to_id=rel_vocab,
            create_inverse_triples=create_inverse_triples
        )

        # Really val on train graph (even under old setting...)
        # But we'll keep naming convention for convenience
        self.inductive_val = TriplesFactory(
            triples=np.array(valid_trips, dtype=str),
            entity_to_id=ent_vocab,
            relation_to_id=rel_vocab,
        )

        self.all_inf_graphs = []
        self.all_inf_test = []
        for i in range(num_test):
            a = TriplesFactory(
                triples=np.array(all_inf_trips[i], dtype=str),
                entity_to_id=inf_ent_vocabs[i],
                relation_to_id=rel_vocab,
                create_inverse_triples=create_inverse_triples
            )
            b = TriplesFactory(
                triples=np.array(all_test_trips[i], dtype=str),
                entity_to_id=inf_ent_vocabs[i],
                relation_to_id=rel_vocab,
            )
            self.all_inf_graphs.append(a)
            self.all_inf_test.append(b)

        # Trust me
        self.transductive_part._num_relations = len(rel_vocab)
        self.inductive_val._num_relations = len(rel_vocab)
        for i in range(num_test):
            self.all_inf_graphs[i]._num_relations = len(rel_vocab)
            self.all_inf_test[i]._num_relations = len(rel_vocab)

   


class InductiveDataset(DataSet):

    def __init__(self,
                 transductive: str,
                 inductive: str,
                 create_inverse_triples: bool = True,
                 cache_root: str = "./data"):

        self.cache_root = cache_root

        self.transductive_part = TriplesFactory(
            path=self.cache_root + transductive + "/train.txt",
            create_inverse_triples=create_inverse_triples
        )
        self.inductive_part = inductive

        self.inductive_inference = TriplesFactory(
            path=self.cache_root + inductive + "/train.txt",
            relation_to_id=self.transductive_part.relation_to_id,
            create_inverse_triples=create_inverse_triples
        )

        self.inductive_val = TriplesFactory(
            path=self.cache_root + inductive + "/valid.txt",
            entity_to_id=self.inductive_inference.entity_to_id,
            relation_to_id=self.transductive_part.relation_to_id
        )

        self.inductive_test = TriplesFactory(
            path=self.cache_root + inductive + "/test.txt",
            entity_to_id=self.inductive_inference.entity_to_id,
            relation_to_id=self.transductive_part.relation_to_id
        )


class Ind_FB15k237(InductiveDataset):

    def __init__(self,
                 version: int = 1,
                 create_inverse_triples: bool = True,):

        super().__init__(transductive=f"fb237_v{version}",
                         inductive=f"fb237_v{version}_ind",
                         create_inverse_triples=create_inverse_triples)


class Ind_WN18RR(InductiveDataset):

    def __init__(self,
                 version: int = 1,
                 create_inverse_triples: bool = True, ):
        super().__init__(transductive=f"WN18RR_v{version}",
                         inductive=f"WN18RR_v{version}_ind",
                         create_inverse_triples=create_inverse_triples)


class Ind_NELL(InductiveDataset):

    def __init__(self,
                 version: int = 1,
                 create_inverse_triples: bool = True, ):
        super().__init__(transductive=f"nell_v{version}",
                         inductive=f"nell_v{version}_ind",
                         create_inverse_triples=create_inverse_triples)
