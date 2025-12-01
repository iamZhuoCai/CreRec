import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from Modules_ori import *
import random

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logging.getLogger().setLevel(logging.INFO)

# torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CreRec-SASRec.")
    parser.add_argument('--random_seed', type=int, default=2025,
                        help='random seed')
    parser.add_argument('--data', nargs='?', default='PolitiFact',
                        help='Politi, Gossip, MHMisinfo_Full')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--model_name', type=str, default='CreRec',
                        help='model name.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-5,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0.95,
                        help='alpha. Politi: 0.95, Gossip: 0.8, MHMisinfo: 0.01')
    parser.add_argument('--beta', type=float, default=5,
                        help='beta. Politi: 5, Gossip: 5, MHMisinfo: 1')
    parser.add_argument('--r', type=float, default=0.2,
                        help='relearning selection ratio. Politi: 0.2, Gossip: 0.4, MHMisinfo: 0.1')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    return parser.parse_args()


args = parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


setup_seed(args.random_seed)

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, fake_items, true_news, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)

        self.feature_transformation = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
        )

        data_directory = './data/' + args.data
        news_embedding_file = os.path.join(data_directory, 'news_embeddings.npy')
        news_embedding_matrix = np.load(news_embedding_file)
        num_embeddings, embedding_dim = news_embedding_matrix.shape
        self.item_embeddings_raw = nn.Embedding(num_embeddings, embedding_dim)
        news_embeddings = torch.from_numpy(news_embedding_matrix).to(device)
        self.item_embeddings_raw.weight.data.copy_(news_embeddings)
        self.item_embeddings_raw.weight.requires_grad = False
        news_embeddings = self.item_embeddings_raw.weight
        news_embeddings3 = self.feature_transformation(news_embeddings)

        self.item_embeddings = nn.Embedding(num_embeddings + 1, hidden_size)

        self.item_embeddings.weight.data.copy_(torch.cat([torch.zeros(1, hidden_size), news_embeddings3], 0))

        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )

        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = ff_out[:, -1, :]


        h = state_hidden.squeeze()
        supervised_output = torch.matmul(h, self.item_embeddings.weight.transpose(0, 1))
        return supervised_output, h

    def forward_eval(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = ff_out[:, -1, :]

        h = state_hidden.squeeze()
        supervised_output = torch.matmul(h, self.item_embeddings.weight.transpose(0, 1))
        return supervised_output
    
    def forward_sequence_embs(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = ff_out[:, -1, :]

        h = state_hidden.squeeze()
        return h

    def forward_sequence_embs_unlearn(self, states, len_states, item_embs):
        inputs_emb = item_embs[states]
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, 0).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = ff_out[:, -1, :]

        h = state_hidden.squeeze()
        return h


def evaluate(model, test_data, true_news, device):
    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    total_purchase = 0.0
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    mrr_purchase = [0, 0, 0, 0]
    true_rate = [0, 0, 0, 0]

    seq, len_seq, target = list(eval_data['seq']), list(eval_data['len_seq']), list(eval_data['next'])

    num_total = len(seq)

    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1) * batch_size], len_seq[i * batch_size: (
                                                                                                                    i + 1) * batch_size], target[
                                                                                                                                          i * batch_size: (
                                                                                                                                                                      i + 1) * batch_size]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)



        prediction = model.forward_eval(states, np.array(len_seq_b))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2 = np.flip(topK, axis=1)
        calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase, mrr_purchase, true_rate, true_news)

        total_purchase += batch_size

    print('#############################################################')

    hr_list = []
    ndcg_list = []
    mrr_list = []
    true_rate_list = []

    header_msg = '{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}\t{:<s}'.format(
            'HR@' + str(topk[0]), 'HR@' + str(topk[1]), 'HR@' + str(topk[2]), 'HR@' + str(topk[3]),
            'NDCG@' + str(topk[0]), 'NDCG@' + str(topk[1]), 'NDCG@' + str(topk[2]), 'NDCG@' + str(topk[3]),
            'MRR@' + str(topk[0]), 'MRR@' + str(topk[1]), 'MRR@' + str(topk[2]), 'MRR@' + str(topk[3]),
            'CR@' + str(topk[0]), 'CR@' + str(topk[1]), 'CR@' + str(topk[2]), 'CR@' + str(topk[3]))
    print(header_msg)

    # logging.info('#############################################################')
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / num_total
        ng_purchase = ndcg_purchase[i] / num_total
        mr_purchase = mrr_purchase[i] / num_total
        tr_rate = true_rate[i] / num_total

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
        true_rate_list.append(tr_rate)


        if i == 1:
            hr_20 = hr_purchase

    results_msg = '{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}\t{:<.4f}'.format(hr_list[0], hr_list[1], hr_list[2], hr_list[3], (ndcg_list[0]), (ndcg_list[1]), (ndcg_list[2]), (ndcg_list[3]), (mrr_list[0]), (mrr_list[1]), (mrr_list[2]), (mrr_list[3]), true_rate_list[0],  true_rate_list[1],
                                                                  true_rate_list[2], true_rate_list[3])
    print(results_msg)
    print('#############################################################')

    return hr_20

if __name__ == '__main__':

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)


    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items

    topk = [5, 10, 20, 50]

    credible_file = os.path.join(data_directory, 'Credible_items.npy')
    credible_items = np.load(credible_file)
    credible_items = credible_items[credible_items !=0]

    all_items = list(range(item_num + 1))
    uncredible_items = [item for item in all_items if item not in credible_items]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device, uncredible_items, credible_items)
    checkpoint = torch.load('./Backbones/pt/SASRec/{}/SASRec.pt'.format(args.data), map_location=device)
    model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), eps=1e-8, weight_decay=args.l2_decay)


    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))

    num_rows = train_data.shape[0]
    # num_batches = int(num_rows / args.batch_size)

    hr_best = 0.0
    tolerate = 0

    unlearning_sequences = train_data[
        (train_data['label_next'] == 1)
    ]


    relearning_sequences = train_data[
        train_data['label_next'] == 0]


    print(unlearning_sequences.shape[0])



    # all training sequences
    batch = train_data.sample(n=train_data.shape[0]).to_dict()
    seq = list(batch['seq'].values())
    len_seq = list(batch['len_seq'].values())
    target = list(batch['next'].values())

    seq = torch.LongTensor(seq)
    len_seq = torch.LongTensor(len_seq)
    target = torch.LongTensor(target)

    seq = seq.to(device)
    target = target.to(device)
    len_seq = len_seq.to(device)

    sequence_embs = model.forward_sequence_embs(seq, len_seq)

    # empirical loss of each interaction sequence
    scores_all = torch.sum(sequence_embs * model.item_embeddings.weight[target], 1)

    # construction of unlearning objective
    uncredible_emb = model.item_embeddings.weight[uncredible_items]  # [n_fake, d]
    credible_emb = model.item_embeddings.weight[credible_items]   # [n_true, d]

    uncredible_emb_ori = uncredible_emb.clone()

    Q, R = torch.linalg.qr(credible_emb.T, mode='reduced')

    E_hat = uncredible_emb_ori - args.alpha * (uncredible_emb_ori @ Q) @ Q.T

    item_embs = model.item_embeddings.weight.clone()
    item_embs[uncredible_items] = E_hat
    

    # unlearning objective
    batch = unlearning_sequences.sample(n=unlearning_sequences.shape[0]).to_dict()
    seq = list(batch['seq'].values())
    len_seq = list(batch['len_seq'].values())
    target = list(batch['next'].values())

    seq = torch.LongTensor(seq)
    len_seq = torch.LongTensor(len_seq)
    target = torch.LongTensor(target)

    seq = seq.to(device)
    target = target.to(device)
    len_seq = len_seq.to(device)

    sequence_embs = model.forward_sequence_embs(seq, len_seq)
    scores_unlearn = torch.sum(sequence_embs * item_embs[target], 1)


    # relearning objective
    batch = relearning_sequences.sample(n=relearning_sequences.shape[0]).to_dict()
    seq = list(batch['seq'].values())
    len_seq = list(batch['len_seq'].values())
    target = list(batch['next'].values())

    seq = torch.LongTensor(seq)
    len_seq = torch.LongTensor(len_seq)
    target = torch.LongTensor(target)

    seq = seq.to(device)
    target = target.to(device)
    len_seq = len_seq.to(device)

    sequence_embs_relearn = model.forward_sequence_embs(seq, len_seq)

    scores_relearn = torch.sum(sequence_embs_relearn * model.item_embeddings.weight[target], 1)

    pos_labels = torch.ones(scores_unlearn.size(0), dtype=scores_unlearn.dtype, device=scores_unlearn.device)

    pos_loss = bce_loss(scores_unlearn.view(-1), pos_labels)

    params_to_grad = [p for n, p in model.named_parameters() if p.requires_grad and ('item_embedding' in n or 'item_embeddings' in n)]
    grads = torch.autograd.grad(pos_loss, params_to_grad, create_graph=True, retain_graph=True, allow_unused=True)

    pos_labels_relearn = torch.ones_like(scores_relearn)
    bce_loss_each = torch.nn.BCEWithLogitsLoss(reduction='none')
    losses_relearn = bce_loss_each(scores_relearn.view(-1), pos_labels_relearn)

    # find bottom r*100% losses
    k = max(1, int(losses_relearn.size(0) * args.r))

    bottomk_vals, bottomk_idx = torch.topk(losses_relearn, k, largest=False)

    scores_relearn_bottomk = scores_relearn[bottomk_idx]

    pos_labels_re = torch.ones(scores_relearn_bottomk.size(0), dtype=scores_relearn_bottomk.dtype, device=scores_relearn_bottomk.device)
    pos_loss_re = bce_loss(scores_relearn_bottomk.view(-1), pos_labels_re)

    # unlearn and relearn on item embeddings
    params_to_grad_re = [p for n, p in model.named_parameters() if p.requires_grad and ('item_embedding' in n or 'item_embeddings' in n)]
    grads_re = torch.autograd.grad(pos_loss_re, params_to_grad_re, create_graph=True, retain_graph=True, allow_unused=True)

    filtered_params = []
    filtered_grads = []
    filtered_params_re = []
    filtered_grads_re = []

    for param, grad in zip(params_to_grad, grads):
        if grad is not None:
            filtered_params.append(param)
            filtered_grads.append(grad)

    for param, grad in zip(params_to_grad_re, grads_re):
        if grad is not None:
            filtered_params_re.append(param)
            filtered_grads_re.append(grad)

    params_to_grad = filtered_params
    grads = filtered_grads

    params_to_grad_re = filtered_params_re
    grads_re = filtered_grads_re

    print(f"Filtered {len(filtered_params)} parameters with non-None gradients")

    # HVP
    def hessian_vector_product(loss, params, v):
        grad1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        grad_v = 0
        for g, vi in zip(grad1, v):
            if g is not None and vi is not None:
                grad_v += (g * vi).sum()
        hv = torch.autograd.grad(grad_v, params, retain_graph=True, allow_unused=True)

        hv_filtered = []
        for h, p in zip(hv, params):
            if h is not None:
                hv_filtered.append(h)
            else:
                hv_filtered.append(torch.zeros_like(p))
        return hv_filtered

    def get_inverse_hvp_cg(loss, params, v, cg_steps=10, damp=1e-2):
        ihvp = [torch.zeros_like(param) for param in params]
        r = [vi.clone() for vi in v]
        p = [vi.clone() for vi in r]
        hv = hessian_vector_product(loss, params, p)
        for i in range(cg_steps):
            # 计算（p, Hv）
            numerator = sum([(ri**2).sum() for ri in r])
            hv = hessian_vector_product(loss, params, p)
            hv_damp = [hvi + damp * pi for hvi, pi in zip(hv, p)]
            denominator = sum([(pi * hvi).sum() for pi, hvi in zip(p, hv_damp)])
            alpha = numerator / (denominator + 1e-10)
            ihvp = [ih + alpha * pi for ih, pi in zip(ihvp, p)]
            r_new = [ri - alpha * hvi for ri, hvi in zip(r, hv_damp)]
            beta = sum([(rni**2).sum() for rni in r_new]) / (numerator + 1e-10)
            p = [rni + beta * pi for rni, pi in zip(r_new, p)]
            r = r_new
        return ihvp

    model.zero_grad()
    all_labels = torch.ones(scores_all.size(0), dtype=scores_all.dtype, device=scores_all.device)
    all_loss = bce_loss(scores_all.view(-1), all_labels)

    inverse_hvp = get_inverse_hvp_cg(all_loss, params_to_grad, grads)

    inverse_hvp_re = get_inverse_hvp_cg(all_loss, params_to_grad, grads_re)

    # unlearn the uncredible influence
    with torch.no_grad():
        for param, ihvpi in zip(params_to_grad, inverse_hvp):
            param += ihvpi
    print('Influence function removal of unlearn objective.')

    # relearn the beneficial influence of relearning objective
    with torch.no_grad():
        for param, ihvpi in zip(params_to_grad, inverse_hvp_re):
            # param -= 1/train_data.shape[0] * ihvpi
            param -= args.beta * ihvpi
    print('Influence function addition of relearning objective.')

    _ = evaluate(model, 'test_data.df', credible_items, device)







