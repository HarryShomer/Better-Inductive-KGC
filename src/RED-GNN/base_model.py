import torch
import numpy as np
import time

from tqdm import tqdm

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import RED_GNN_induc
# from models_old import RED_GNN_induc
from utils import cal_ranks, cal_performance

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = RED_GNN_induc(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

    def train_batch(self, epoch):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        t_time = time.time()
        self.model.train()
        for i in tqdm(range(n_batch), "Training"):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:,0], triple[:,1])

            pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(triple[:,2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()

        self.scheduler.step()
        self.t_time += time.time() - t_time

        if epoch % 2 == 0 and epoch > 0:
            valid_mrr, out_str = self.evaluate()
        else:
            valid_mrr, out_str = -1, ""

        return valid_mrr, out_str
    

    def evaluate(self, ):
        batch_size = self.n_batch

        # n_data = self.n_valid
        # n_batch = n_data // batch_size + (n_data % batch_size > 0)
        # ranking = []
        # self.model.eval()
        i_time = time.time()
        # for i in tqdm(range(n_batch), "Valid"):
        #     start = i*batch_size
        #     end = min(n_data, (i+1)*batch_size)
        #     batch_idx = np.arange(start, end)
        #     subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
        #     scores = self.model(subs, rels).data.cpu().numpy()
        #     filters = []
        #     for i in range(len(subs)):
        #         filt = self.loader.val_filters[(subs[i], rels[i])]
        #         filt_1hot = np.zeros((self.n_ent, ))
        #         filt_1hot[np.array(filt)] = 1
        #         filters.append(filt_1hot)
             
        #     filters = np.array(filters)
        #     ranks = cal_ranks(scores, objs, filters)
        #     ranking += ranks
        # ranking = np.array(ranking)
        # v_mrr, v_h1, v_h10 = cal_performance(ranking)

        # out_str = f'[VALID] MRR:{v_mrr:.4f} H@1:{v_h1:.4f} H@10:{v_h10:.4f}'
        out_str = " "
        v_mrr = -1

        self.model.eval()
        for j in range(self.loader.num_test):
            n_data = self.n_test[j]
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []

            for i in tqdm(range(n_batch), f"Test Graph {j}"):
                start = i*batch_size
                end = min(n_data, (i+1)*batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data='test', test_graph=j)
                scores = self.model(subs, rels, 'inductive', test_graph=j).data.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = self.loader.tst_filters[j][(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent_ind[j], ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
             
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks
            
            # Done for each test graph individually
            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            out_str += f" [TEST {j}] MRR:{t_mrr:.4f} H@1:{t_h1:.4f} H@10:{t_h10:.4f}"

        i_time = time.time() - i_time
        out_str = out_str + "\n"

        return v_mrr, out_str

