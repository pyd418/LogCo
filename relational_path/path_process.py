import numpy as np
import itertools
import os
import dgl
import torch
import torch.nn as nn
from relational_path.path_sampler import obtain_dglpaths, obtain_batch_dgl_rel_paths, unfold_rel_paths, create_neg_paths


def path_generate(data, max_path_len, rels_num):
    g, rel_labels = data

    graphs_list = list(dgl.unbatch(g))
    # rel_temp = graphs_list[10].edata['type'][graphs_list[10].edge_id(0, 1)]
    root_list = np.zeros(len(graphs_list), dtype=int)
    end_list = np.ones(len(graphs_list), dtype=int)
    hop_list = np.ones(len(graphs_list), dtype=int) * max_path_len

    # entity paths
    param_ent = zip(graphs_list, root_list, end_list, hop_list)
    entity_paths = list(itertools.starmap(obtain_dglpaths, list(param_ent)))

    # unfold relation paths
    param_rel = zip(entity_paths, graphs_list)
    rel_paths = list(itertools.starmap(obtain_batch_dgl_rel_paths, list(param_rel)))
    rel_path_batch = list(map(unfold_rel_paths, rel_paths))

    # generate negative relation paths
    rel_list = np.ones(len(graphs_list), dtype=int) * rels_num
    param_neg = zip(rel_path_batch, rel_labels, rel_list)
    neg_paths_batch = list(itertools.starmap(create_neg_paths, list(param_neg)))

    return rel_path_batch, neg_paths_batch, rel_labels

def path_generate_mul_neg(data, max_path_len, rels_num):
    g, rel_labels = data

    graphs_list = list(dgl.unbatch(g))
    root_list = np.zeros(len(graphs_list), dtype=int)
    end_list = np.ones(len(graphs_list), dtype=int)
    hop_list = np.ones(len(graphs_list), dtype=int) * max_path_len

    # entity paths
    param_ent = zip(graphs_list, root_list, end_list, hop_list)
    entity_paths = list(itertools.starmap(obtain_dglpaths, list(param_ent)))

    # unfold relation paths
    param_rel = zip(entity_paths, graphs_list)
    rel_paths = list(itertools.starmap(obtain_batch_dgl_rel_paths, list(param_rel)))
    rel_path_batch = list(map(unfold_rel_paths, rel_paths))

    # generate negative relation paths
    rel_list = np.ones(len(graphs_list), dtype=int) * rels_num
    param_neg = zip(rel_path_batch, rel_labels, rel_list)
    neg_paths_batch = list(itertools.starmap(create_neg_paths, list(param_neg)))
    param_neg_2 = zip(rel_path_batch, rel_labels, rel_list)

    neg_paths_batch_2 = list(itertools.starmap(create_neg_paths, list(param_neg_2)))
    for i in range(0, len(neg_paths_batch)):
        neg_paths_batch[i].extend(neg_paths_batch_2[i])

    return rel_path_batch, neg_paths_batch, rel_labels

def get_paths_nums(rel_path_batch):
    num_list = list(map(len, rel_path_batch))
    paths_nums = np.mean(num_list)
    return paths_nums


def path_emb_generate_batch(rel_path_batch, rel_emb, target_labels, epoch, max_epoch, filename, pos):
    s_p_batch = []
    paths_emb_batch = []
    # alpha_p = []
    label_id = 0
    # batch_size = len(rel_paths_batch)
    for paths_in_graph in rel_path_batch:
        paths_emb, score_paths = path_emb_generate_rnn(paths_in_graph, rel_emb, target_labels[label_id])
        s_p = torch.matmul(score_paths.squeeze(0), torch.Tensor(paths_emb))    # Tensor([0.1, 0.2, ...])
        s_p_batch.append(s_p.tolist())      # [[0.1, 0.2, ...],[0.1, 0.2, ...],[0.1, 0.2, ...]...]
        paths_emb_batch.append(paths_emb)      
        if epoch == max_epoch:
            if pos == 1:
                print_attetion(paths_in_graph, score_paths.squeeze(0), target_labels[label_id], filename)
        label_id += 1
    s_p_batch = torch.Tensor(s_p_batch)     # Tensor([[0.1, 0.2, ...],[0.1, 0.2, ...],[0.1, 0.2, ...]...])
    return paths_emb_batch, s_p_batch


def print_attetion(paths_in_graph, score_paths, target_rel, file_name):
    data_path = os.path.join(file_name, f"target_{target_rel}/alpha.txt")
    dir_path = os.path.join(file_name, f"target_{target_rel}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(data_path, 'a') as f:
        for i in range(len(paths_in_graph)):
            f.write(str(paths_in_graph[i]) + str(score_paths[i]) + '\n')
        f.writelines('\n')
    # return label_id


def path_emb_generate_batch_ori(rel_path_batch, rel_emb, target_labels):
    s_p_batch = []
    paths_emb_batch = []
    # alpha_p = []
    label_id = 0
    # batch_size = len(rel_paths_batch)
    for paths_in_graph in rel_path_batch:
        paths_emb, score_paths = path_emb_generate(paths_in_graph, rel_emb, target_labels[label_id])
        s_p = torch.matmul(score_paths.squeeze(0), torch.Tensor(paths_emb))    # Tensor([0.1, 0.2, ...])
        s_p_batch.append(s_p.tolist())      # [[0.1, 0.2, ...],[0.1, 0.2, ...],[0.1, 0.2, ...]...]
        paths_emb_batch.append(paths_emb)     
        label_id += 1
    s_p_batch = torch.Tensor(s_p_batch)     # Tensor([[0.1, 0.2, ...],[0.1, 0.2, ...],[0.1, 0.2, ...]...])
    return paths_emb_batch, s_p_batch


def path_emb_generate_rnn(paths, rel_emb, target_id):    # list tensor int
    paths_emb = []  # list
    score_paths = torch.Tensor([])    # list

    conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2)

    for path in paths:  # path: list
        # index = np.asarray(path)
        # index = np.array(path, dtype=int)
        # index = torch.LongTensor([0, 1])
        # rnn = nn.LSTM(32, 32, num_layers=2)    
        # input_rnn = rel_emb(torch.LongTensor(path))
        # output_rnn, (h, c) = rnn(input_rnn.unsqueeze(dim=0))
        if len(path) < 2:
            path_emb = rel_emb(torch.LongTensor(path))
        else:
            input_rnn = rel_emb(torch.LongTensor(path))
            input_rnn = input_rnn.unsqueeze(dim=0).permute(0, 2, 1)
            path_emb = conv1(input_rnn)
            path_emb = path_emb.permute(0, 1, 2).squeeze(0)
            path_emb = torch.sum(path_emb, dim=1)
            # print(path_emb.shape)
        path_emb = path_emb.squeeze(0).tolist()
        # print(output_rnn.squeeze(0).shape)
        # path_emb = torch.sum(output_rnn.squeeze(0), dim=0).tolist()  # tensor ->tolist
        # path_emb = torch.mean(rel_emb[index], dim=1)
        # print(torch.Tensor(path_emb).shape)
        paths_emb.append(path_emb)  # list: [[0.1, 0.2, ...],[0.1, 0.2, ...],[0.1, 0.2, ...]...]
        # print(path_emb.shape)
        
        score = torch.dot(torch.Tensor(path_emb), rel_emb(target_id))       
        # score_paths.append(score)
        score_paths = torch.cat((score_paths, score.view(1, 1)), dim=0)     # Tensor([])
    softmax = nn.Softmax(dim=1)
    alpha_paths = softmax(score_paths.view(1, len(score_paths)))            # Tensor([[0.1, 0.2, ...]])
    return paths_emb, alpha_paths


def path_emb_generate(paths, rel_emb, target_id):    # list tensor int
    paths_emb = []  # list
    score_paths = torch.Tensor([])    # list
    for path in paths:  # path: list
        # index = np.asarray(path)
        # index = np.array(path, dtype=int)
        # index = torch.LongTensor([0, 1])
        # print(rel_emb(torch.LongTensor(path)).shape)
        path_emb = torch.sum(rel_emb(torch.LongTensor(path)), dim=0).tolist()  # tensor ->tolist
        # print(path_emb.shape)
        # path_emb = torch.mean(rel_emb(torch.LongTensor(path)), dim=0).tolist()  # tensor ->tolist
        # path_emb = torch.mean(rel_emb[index], dim=1)
        paths_emb.append(path_emb)  # list: [[0.1, 0.2, ...],[0.1, 0.2, ...],[0.1, 0.2, ...]...]
       
        score = torch.dot(torch.Tensor(path_emb), rel_emb(target_id))       
        # score_paths.append(score)
        score_paths = torch.cat((score_paths, score.view(1, 1)), dim=0)     # Tensor([])
    softmax = nn.Softmax(dim=1)
    alpha_paths = softmax(score_paths.view(1, len(score_paths)))            # Tensor([[0.1, 0.2, ...]])
    return paths_emb, alpha_paths


def path_cross_loss(s_p_batch, rel_emb, target_labels):
    # index = np.array(target_labels, dtype=int)
    # target_embs = torch.Tensor(rel_emb[index])
    # score = torch.sum(torch.mul(torch.Tensor(s_p_batch), target_embs), dim=1)   # [1, 2, 3, ...]
    output = torch.matmul(s_p_batch, rel_emb.t())
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(output, target_labels)
    return loss


def path_contrast_loss(s_p_pos, s_p_neg, rel_emb, target_labels):
    batch_size = len(target_labels)
    pos = torch.mul(s_p_pos, rel_emb(target_labels))
    neg = torch.mul(s_p_neg, rel_emb(target_labels))
    softmax = nn.Softmax(dim=0)
    pos_sim = torch.sum(pos, dim=1)    # tensor
    neg_sim = torch.sum(neg, dim=1)    # tensor

    sim = torch.cat((pos_sim.view(1, len(pos_sim)), neg_sim.view(1, len(neg_sim))), dim=0)
    output = softmax(sim)

    zero_id = (output[0] == 0).nonzero(as_tuple=False).flatten()
    output[0][zero_id] = output[0][zero_id] + 1e-10
    temp = output[0]
    # numerator = torch.exp(pos_sim)
    # denominator = torch.exp(pos_sim) + torch.exp(neg_sim)
    res = - torch.log(output[0])   # tensor, triple?
    # loss = torch.sum(res) / (2 * batch_size)
    loss = torch.sum(res)
    return loss


def path_contrast_loss_2(pos_paths_emb_batch, neg_paths_emb_batch, rel_emb, target_labels, device):
    batch_size = len(target_labels)
    label_id = 0
    loss_batch = 0
    softmax = nn.Softmax(dim=0)
    for paths_emb in zip(pos_paths_emb_batch, neg_paths_emb_batch):
        pos_sim = torch.matmul(torch.Tensor(paths_emb[0]).to(device=device), rel_emb(target_labels[label_id]))     
        neg_sim = torch.matmul(torch.Tensor(paths_emb[1]).to(device=device), rel_emb(target_labels[label_id]))

        sim = torch.cat((pos_sim.view(1, len(pos_sim)), neg_sim.view(1, len(neg_sim))), dim=0)
        output = softmax(sim)

        zero_id = (output[0] == 0).nonzero(as_tuple=False).flatten()
        output[0][zero_id] = output[0][zero_id] + 1e-10
        res = - torch.log(output[0])  # tensor, triple?
        # loss = torch.sum(res) / (2 * batch_size)
        loss = torch.sum(res)
        label_id += 1
        loss_batch = loss_batch + loss

    return loss_batch
