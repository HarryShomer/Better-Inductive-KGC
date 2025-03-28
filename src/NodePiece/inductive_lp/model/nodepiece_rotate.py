import torch
import random
import numpy as np
from torch.nn import functional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional
from collections import Counter, defaultdict

from torch import nn
from pykeen.losses import Loss
from pykeen.triples import TriplesFactory
from pykeen.models import Model
from tqdm import tqdm
from torch_geometric.data import Data

from .comp_gcn import StarE_PyG_Encoder


class NodePieceRotate(Model):

    def __init__(self,
                 tokenizer = None,
                 triples: TriplesFactory = None,
                 inference_triples: TriplesFactory = None,
                 device: torch.device = None,
                 loss: Loss = None,
                 max_paths: int = None,  # max anchors per node
                 subbatch: int = 32,
                 max_seq_len: int = None,  # tied with anchor distances
                 embedding_dim: int = 100,
                 hid_dim: int = 200,  # hidden dim for the hash encoder
                 num_heads: int = 4,  # for Trf
                 num_layers: int = 2,  # for Trf
                 pooler: str = "cat",  # "cat" or "trf"
                 drop_prob: float = 0.1,  # dropout
                 use_distances: bool = True,
                 rel_policy: str = "sum",
                 random_hashes: int = 0,  # for ablations
                 nearest: bool = True,  # use only K nearest anchors per node
                 sample_rels: int = 0,  # size of the relational context
                 ablate_anchors: bool = False,  # for ablations - node hashes will be constructed only from the relational context
                 tkn_mode: str = "path",  # default NodePiece vocabularization strategy
                 gnn: bool = False,
                 gnn_att: bool = False,
                 gcn_att_heads: int = 2,
                 use_lrga: bool = False,
                 gnn_layers: int = 2,
                 gnn_att_drop: float = 0.1,
                 pna: bool = False,
                 residual: bool = False,
                 jk: bool = False,
                 ):

        super().__init__(
            triples_factory=inference_triples,
            loss=loss,
            predict_with_sigmoid=False,
            automatic_memory_optimization=False,
            preferred_device=device,
        )

        self.pooler = pooler
        self.policy = rel_policy
        self.nearest = nearest
        self.sample_rels = sample_rels
        self.ablate_anchors = ablate_anchors
        self.tkn_mode = tkn_mode
        self.gnn = gnn


        # cat pooler - concat all anchors+relations in one big vector, pass through a 2-layer MLP
        if pooler == "cat":
            self.set_enc = nn.Sequential(
                nn.Linear(embedding_dim * (max_paths + sample_rels), embedding_dim * 2), nn.Dropout(drop_prob), nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            ) if not self.ablate_anchors else nn.Sequential(
                nn.Linear(embedding_dim * sample_rels, embedding_dim * 2), nn.Dropout(drop_prob), nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
        # trf pooler - vanilla transformer encoder with mean pooling on top
        elif pooler == "trf":
            encoder_layer = TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hid_dim,
                dropout=drop_prob,
            )
            self.set_enc = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)



        self.device = device
        self.loss = loss
        self.transductive_triples_factory = triples
        self.inductive_inference = inference_triples
        #self.tokenizer = tokenizer
        self.random_hashes = random_hashes

        self.subbatch = subbatch
        self.embedding_dim = embedding_dim
        self.real_embedding_dim = embedding_dim // 2  # RotatE interaction assumes vectors conists of two parts: real and imaginary

        self.max_seq_len = max_seq_len
        self.sample_paths = max_paths
        self.use_distances = use_distances

        # pykeen stuff
        self.automatic_memory_optimization = False

        # self.tokenizer.token2id[self.tokenizer.NOTHING_TOKEN] = len(tokenizer.token2id) - 1  # TODO this is a bugfix as PathTrfEncoder puts its own index here
        # self.anchor_embeddings = nn.Embedding(len(tokenizer.token2id), embedding_dim=embedding_dim, padding_idx=self.tokenizer.token2id[tokenizer.PADDING_TOKEN])

        self.relation_embeddings = nn.Embedding(self.transductive_triples_factory.num_relations + 1, embedding_dim=embedding_dim, padding_idx=self.transductive_triples_factory.num_relations)
        # self.dist_emb = nn.Embedding(self.max_seq_len, embedding_dim=embedding_dim)
        self.entity_embeddings = None

        # now fix anchors per node for each node in a graph, either deterministically or randomly
        # we do it mostly for speed reasons, although this process can happen during the forward pass either
        if not self.ablate_anchors:
            if self.random_hashes == 0:
                if self.tkn_mode != "bfs":
                    # DETERMINISTIC strategy
                    if not self.nearest:
                        # subsample paths, need to align them with distances
                        sampled_paths = {
                            entity: random.sample(paths, k=min(self.sample_paths, len(paths)))
                            for entity, paths in self.tokenizer.vocab.items()
                        }
                    elif self.nearest:
                        # sort paths by length first and take K of them
                        sampled_paths = {
                            entity: sorted(paths, key=lambda x: len(x))[:min(self.sample_paths, len(paths))]
                            for entity, paths in self.tokenizer.vocab.items()
                        }
                        self.max_seq_len = max(len(path) for k,v in sampled_paths.items() for path in v)
                        print(f"Changed max seq len from {max_seq_len} to {self.max_seq_len} after keeping {self.sample_paths} shortest paths")

                    hashes = [
                        [self.tokenizer.token2id[path[0]] for path in paths] + [self.tokenizer.token2id[tokenizer.PADDING_TOKEN]]*(self.sample_paths - len(paths))
                        for entity, paths in sampled_paths.items()
                    ]
                    distances = [
                        [len(path)-1 for path in paths] + [0] *(self.sample_paths - len(paths))
                        for entity, paths in sampled_paths.items()
                    ]
                    total_paths = [
                        [len(paths)]
                        for entity, paths in sampled_paths.items()
                    ]
                else:
                    # only nearest neighbors mode
                    if not self.nearest:
                        raise NotImplementedError("bfs mode works only with -nn True")
                    hashes = [
                        [self.tokenizer.token2id[token] for token in vals['ancs'][:min(self.sample_paths, len(vals['ancs']))]] + [self.tokenizer.token2id[tokenizer.PADDING_TOKEN]] * (self.sample_paths - len(vals['ancs']))
                        for entity, vals in self.tokenizer.vocab.items()
                    ]
                    distances = [
                        [d for d in vals['dists'][:min(self.sample_paths, len(vals['dists']))]] + [0] * (self.sample_paths - len(vals['dists']))
                        for entity, vals in self.tokenizer.vocab.items()
                    ]
                    total_paths = distances
                    self.max_seq_len = max([d for row in distances for d in row])
                    print(f"Changed max seq len from {max_seq_len} to {self.max_seq_len} after keeping {self.sample_paths} shortest paths")

                self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
                self.distances = torch.tensor(distances, dtype=torch.long, device=self.device)
                self.total_paths = torch.tensor(total_paths, dtype=torch.long, device=self.device)

            else:
                # RANDOM strategy
                # in this case, we bypass distances and won't use relations in the encoder
                self.anchor_embeddings = nn.Embedding(self.random_hashes, embedding_dim=embedding_dim)
                hashes = [
                    random.sample(list(range(random_hashes)), self.sample_paths)
                    for i in range(triples.num_entities)
                ]

                self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
                self.distances = torch.zeros((triples.num_entities, self.sample_paths), dtype=torch.long, device=self.device)
                self.total_paths = torch.zeros((triples.num_entities, 1), dtype=torch.long, device=self.device)

        # creating the relational context of M unique outgoing relation types for each node
        if self.sample_rels > 0:
            pad_idx = self.transductive_triples_factory.num_relations
            self.pad_idx = pad_idx
            e2r = defaultdict(set)
            ind_e2r = defaultdict(set)
            for row in self.transductive_triples_factory.mapped_triples:
                e2r[row[0].item()].add(row[1].item())

            for row in self.inductive_inference.mapped_triples:
                ind_e2r[row[0].item()].add(row[1].item())

            len_stats = [len(v) for k,v in e2r.items()]
            len_ind_stats = [len(v) for k,v in ind_e2r.items()]
            print(f"Transductive: Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(len_stats, 66)}, max: {max(len_stats)} ")
            print(f"Inductive: Unique relations per node - min: {min(len_ind_stats)}, avg: {np.mean(len_ind_stats)}, 66th perc: {np.percentile(len_ind_stats, 66)}, max: {max(len_ind_stats)} ")
            unique_1hop_relations = [
                random.sample(e2r[i], k=min(self.sample_rels, len(e2r[i]))) + [pad_idx] * (self.sample_rels-min(len(e2r[i]), self.sample_rels))
                for i in range(self.transductive_triples_factory.num_entities)
            ]
            self.unique_1hop_relations = torch.tensor(unique_1hop_relations, dtype=torch.long, device=self.device)

            self.inference_1hop_relations = torch.tensor(
                [
                    random.sample(ind_e2r[i], k=min(self.sample_rels, len(ind_e2r[i]))) + [pad_idx] * (
                                self.sample_rels - min(len(ind_e2r[i]), self.sample_rels))
                    for i in range(self.inductive_inference.num_entities)
                ],
                dtype=torch.long,
                device=self.device
            )

        if self.gnn:
            self.gnn_encoder = StarE_PyG_Encoder(
                emb_dim=embedding_dim,
                device=device,
                num_rel=len(self.relation_embeddings.weight),
                use_lrga=use_lrga,
                num_layers=gnn_layers,
                drop1=drop_prob,
                residual=residual,
                jk=jk,
                layer_config={
                    'STATEMENT_LEN': 3,
                    'EMBEDDING_DIM': embedding_dim,
                    'STAREARGS': {
                        'GCN_DROP': drop_prob,
                        'ATTENTION': gnn_att,
                        'GCN_DIM': embedding_dim,
                        'ATTENTION_HEADS': gcn_att_heads,
                        'ATTENTION_SLOPE': 0.1,
                        'ATTENTION_DROP': gnn_att_drop,
                        'BIAS': False,
                        'OPN': 'rotate',
                        'QUAL_AGGREGATE': 'sum',
                        'PNA': pna,
                    },
                },
            )


    def resample_rels(self):
        """
        Note: Change self.inductive_inference first

        This is used in ER setting to change sampling for new inference graph
        """
        print("Resampling Relations...")
        
        pad_idx = self.transductive_triples_factory.num_relations
        self.pad_idx = pad_idx
        e2r = defaultdict(set)
        ind_e2r = defaultdict(set)
        for row in self.transductive_triples_factory.mapped_triples:
            e2r[row[0].item()].add(row[1].item())

        for row in self.inductive_inference.mapped_triples:
            ind_e2r[row[0].item()].add(row[1].item())

        len_stats = [len(v) for k,v in e2r.items()]
        len_ind_stats = [len(v) for k,v in ind_e2r.items()]
        print(f"Transductive: Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(len_stats, 66)}, max: {max(len_stats)} ")
        print(f"Inductive: Unique relations per node - min: {min(len_ind_stats)}, avg: {np.mean(len_ind_stats)}, 66th perc: {np.percentile(len_ind_stats, 66)}, max: {max(len_ind_stats)} ")
        unique_1hop_relations = [
            random.sample(e2r[i], k=min(self.sample_rels, len(e2r[i]))) + [pad_idx] * (self.sample_rels-min(len(e2r[i]), self.sample_rels))
            for i in range(self.transductive_triples_factory.num_entities)
        ]
        self.unique_1hop_relations = torch.tensor(unique_1hop_relations, dtype=torch.long, device=self.device)

        self.inference_1hop_relations = torch.tensor(
            [
                random.sample(ind_e2r[i], k=min(self.sample_rels, len(ind_e2r[i]))) + [pad_idx] * (
                            self.sample_rels - min(len(ind_e2r[i]), self.sample_rels))
                for i in range(self.inductive_inference.num_entities)
            ],
            dtype=torch.long,
            device=self.device
        )

    def _reset_parameters_(self):

        if self.pooler != "avg":
            for module in self.set_enc.modules():
                if module is self:
                    continue
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            if self.pooler == "mlp":
                for module in self.set_dec.modules():
                    if module is self:
                        continue
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

        if not self.ablate_anchors:
            torch.nn.init.xavier_uniform_(self.anchor_embeddings.weight)
            torch.nn.init.xavier_uniform_(self.dist_emb.weight)


            if self.random_hashes == 0:
                with torch.no_grad():
                    self.anchor_embeddings.weight[self.tokenizer.token2id[self.tokenizer.PADDING_TOKEN]] = torch.zeros(self.embedding_dim)
                    self.dist_emb.weight[0] = torch.zeros(self.embedding_dim)


        # for RotatE: phases randomly between 0 and 2 pi
        phases = 2 * np.pi * torch.rand(self.transductive_triples_factory.num_relations, self.real_embedding_dim, device=self.device)
        relations = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1).detach()
        assert torch.allclose(torch.norm(relations, p=2, dim=-1), phases.new_ones(size=(1, 1)))
        self.relation_embeddings.weight.data[:-1] = relations.view(self.transductive_triples_factory.num_relations, self.embedding_dim)
        self.relation_embeddings.weight.data[-1] = torch.zeros(self.embedding_dim)

        if self.gnn:
            self.gnn_encoder.reset_parameters()


    def post_parameter_update(self):  # noqa: D102

        # Make sure to call super first
        super().post_parameter_update()

        # Normalize relation embeddings
        rel = self.relation_embeddings.weight.data.view(self.transductive_triples_factory.num_relations+1, self.real_embedding_dim, 2)
        rel = functional.normalize(rel, p=2, dim=-1)
        self.relation_embeddings.weight.data = rel.view(self.transductive_triples_factory.num_relations+1, self.embedding_dim)

        self.entity_embeddings = None


    def pool_anchors(self, anc_embs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        input shape: (bs, num_anchors + relational_context, emb_dim)
        output shape: (bs, emb_dim)
        """

        if self.pooler == "cat":
            anc_embs = anc_embs.view(anc_embs.shape[0], -1)
            pooled = self.set_enc(anc_embs) if self.sample_paths != 1 else anc_embs
        elif self.pooler == "trf":
            pooled = self.set_enc(anc_embs.transpose(1, 0), src_key_padding_mask=mask)  # output shape: (seq_len, bs, dim)
            pooled = pooled.mean(dim=0)  # output shape: (bs, dim)

        return pooled


    def encode_by_index(self, entities: torch.LongTensor) -> torch.FloatTensor:

        # take a node index and find its NodePiece hash

        # hashes, dists, ids = self.hashes[entities], self.distances[entities], self.total_paths[entities]
        #
        # anc_embs = self.anchor_embeddings(hashes)
        # mask = None
        #
        # if self.use_distances:
        #     dist_embs = self.dist_emb(dists)
        #     anc_embs += dist_embs
        #
        # # for ablations: drop anchors entirely
        # if self.ablate_anchors:
        anc_embs = torch.tensor([], device=self.device)

        # add relational context (if its size > 0 )
        if self.sample_rels > 0:
            if self.training:
                rels = self.unique_1hop_relations[entities]  # (bs, rel_sample_size)
            else:
                rels = self.inference_1hop_relations[entities]
            mask = rels == self.pad_idx
            rels = self.relation_embeddings(rels)   # (bs, rel_sample_size, dim)
            anc_embs = torch.cat([anc_embs, rels], dim=1)  # (bs, ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask.to(anc_embs.device))  # (bs, dim)

        return anc_embs


    def get_all_representations(self):

        # materialize embeddings for all nodes in a graph for scoring


        if self.training:
            num_ents = len(self.unique_1hop_relations)
        else:
            num_ents = len(self.inference_1hop_relations)

        temp_embs = torch.zeros((num_ents, self.embedding_dim), dtype=torch.float, device=self.device)

        vocab_keys = list(range(num_ents))
        for i in range(0, num_ents, self.subbatch):
            entities = torch.tensor(vocab_keys[i: i+self.subbatch], dtype=torch.long, device=self.device)
            embs = self.encode_by_index(entities)
            temp_embs[i: i + self.subbatch, :] = embs

        return temp_embs

    def encode_gnn(self):
        data = Data()
        data['x'] = self.get_all_representations()
        data['rels'] = self.relation_embeddings.weight
        if self.training:
            graph = self.transductive_triples_factory.mapped_triples
        else:
            graph = self.inductive_inference.mapped_triples

        direct_edges = graph[graph[:, 1] % 2 == 0][:, [0,2]].T
        inverse_edges = graph[graph[:, 1] % 2 == 1][:, [0,2]].T

        direct_types = graph[graph[:, 1] % 2 == 0, 1]
        inverse_types = graph[graph[:, 1] % 2 == 1, 1]
        data['edge_index'] = torch.cat([direct_edges, inverse_edges], dim=1)
        data['edge_type'] = torch.cat([direct_types, inverse_types], dim=0)

        x, r = self.gnn_encoder.forward_base(data.to(self.device))

        return x, r

    def encode_gnn_entities(self, heads, rels, tails):
        x, r = self.encode_gnn()
        return x[heads] if heads is not None else x, r[rels], x[tails] if tails is not None else x


    @staticmethod
    def pairwise_interaction_function(
            h: torch.FloatTensor,
            r: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # Decompose into real and imaginary part
        h_re = h[..., 0]
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.cat(
            [
                h_re * r_re - h_im * r_im,
                h_re * r_im + h_im * r_re,
            ],
            dim=-1,
        )

        return rot_h

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # Decompose into real and imaginary part
        h_re = h[..., 0]
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.stack(
            [
                h_re * r_re - h_im * r_im,
                h_re * r_im + h_im * r_re,
            ],
            dim=-1,
        )
        # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
        diff = rot_h - t
        scores = -torch.norm(diff.view(diff.shape[:-2] + (-1,)), dim=-1)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102


        # when training with large # of neg samples the hrt_batch size can be too big to fit into memory, so chunk it
        if hrt_batch.shape[0] <= self.subbatch or self.subbatch == 0:
            # Get embeddings
            if not self.gnn:
                h = self.encode_by_index(hrt_batch[:, 0]).view(-1, self.real_embedding_dim, 2)
                r = self.relation_embeddings(hrt_batch[:, 1]).view(-1, self.real_embedding_dim, 2)
                t = self.encode_by_index(hrt_batch[:, 2]).view(-1, self.real_embedding_dim, 2)
            else:
                h, r, t = self.encode_gnn_entities(hrt_batch[:, 0], hrt_batch[:, 1], hrt_batch[:, 2])
                h, r, t = h.view(-1, self.real_embedding_dim, 2), r.view(-1, self.real_embedding_dim, 2), t.view(-1, self.real_embedding_dim, 2)

            # Compute scores
            scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)
        else:
            scores = torch.zeros((hrt_batch.shape[0], 1), dtype=torch.float, device=hrt_batch.device)
            for i in range(0, hrt_batch.shape[0], self.subbatch):
                if not self.gnn:
                    h = self.encode_by_index(hrt_batch[i: i+self.subbatch, 0]).view(-1, self.real_embedding_dim, 2)
                    r = self.relation_embeddings(hrt_batch[i: i+self.subbatch, 1]).view(-1, self.real_embedding_dim, 2)
                    t = self.encode_by_index(hrt_batch[i: i+self.subbatch, 2]).view(-1, self.real_embedding_dim, 2)
                else:
                    h, r, t = self.encode_gnn_entities(hrt_batch[i: i+self.subbatch, 0], hrt_batch[i: i+self.subbatch, 1], hrt_batch[i: i+self.subbatch, 2])
                    h, r, t = h.view(-1, self.real_embedding_dim, 2), r.view(-1, self.real_embedding_dim, 2), t.view(-1, self.real_embedding_dim, 2)

                # Compute scores
                scores[i: i+self.subbatch] = self.interaction_function(h=h, r=r, t=t).view(-1, 1)


        return scores


    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102

        if not self.gnn:
            # Get embeddings
            h = self.encode_by_index(hr_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
            r = self.relation_embeddings(hr_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

            # Rank against all entities, don't use hard negs, EXPENSIVE
            t = self.get_all_representations().view(1, -1, self.real_embedding_dim, 2)
        else:
            h, r, t = self.encode_gnn_entities(hr_batch[:, 0], hr_batch[:, 1], None)
            h, r, t = h.view(-1, 1, self.real_embedding_dim, 2), r.view(-1, 1, self.real_embedding_dim, 2), t.view(1, -1, self.real_embedding_dim, 2)

        # Compute scores
        if self.subbatch == 0:
            scores = self.interaction_function(h=h, r=r, t=t)
        else:
            scores = torch.zeros((hr_batch.shape[0], t.shape[1]), dtype=torch.float, device=hr_batch.device)
            for i in tqdm(range(0, t.shape[1], self.subbatch)):
                temp_scores = self.interaction_function(h=h, r=r, t=t[:, i:i+self.subbatch, :, :])
                scores[:, i:i+self.subbatch] = temp_scores

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102

        # Get embeddings
        if not self.gnn:
            r = self.relation_embeddings(rt_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
            t = self.encode_by_index(rt_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

            r_inv = torch.stack([r[:, :, :, 0], -r[:, :, :, 1]], dim=-1)

            # Rank against all entities
            h = self.get_all_representations().view(1, -1, self.real_embedding_dim, 2)
        else:
            h, r, t = self.encode_gnn_entities(None, rt_batch[:, 0], rt_batch[:, 1])
            h, r, t = h.view(1, -1, self.real_embedding_dim, 2), r.view(-1, 1, self.real_embedding_dim, 2), t.view(-1, 1, self.real_embedding_dim, 2)
            r_inv = torch.stack([r[:, :, :, 0], -r[:, :, :, 1]], dim=-1)

        # Compute scores
        if self.subbatch == 0:
            scores = self.interaction_function(h=t, r=r_inv, t=h)
        else:
            scores = torch.zeros((rt_batch.shape[0], t.shape[1]), dtype=torch.float, device=rt_batch.device)
            for i in tqdm(range(0, t.shape[1], self.subbatch)):
                temp_scores = self.interaction_function(h=t, r=r_inv, t=h[:, i:i+self.subbatch, :, :])
                scores[:, i:i+self.subbatch] = temp_scores


        return scores