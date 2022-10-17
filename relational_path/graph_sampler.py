import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
from .path_sampler import *
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across relations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    # 进度条
    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def links2subgraphs(A, graphs, params, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM

    # env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)   # 创建db文件
    env = lmdb.open(params.db_path, map_size=int(1e9), max_dbs=6)  # 创建db文件

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            # 进度条, p.map后面是函数名与参数，extract_save_subgraph 抽出子图所有数据
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                # 存入子图数据
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env) # 在这一步抽出路径与路径负例

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    # 随机选择sample_size个样本(triples)，对于每一个triple计算子图
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        # 余下的节点，labels，子图大小，enclosing比例，筛掉的节点个数
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)

        # 此处加入path的key
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_

    # 余下的节点，labels，子图大小，enclosing比例，筛掉的节点个数
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    # 此处如果为pos，加入path的key
    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    # 有向图转化为无向图
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)  # 交集
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)  # 并集

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    # 得到子图邻接矩阵（多个）
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    # 在此处添加构建路径的函数,输入为子图的incidence_matrix, 可以先不着急看边的属性
    ##################################
    # subgraph_incidence = incidence_matrix(subgraph)
    # subgraph_incidence += subgraph_incidence.T

    # paths = find_paths(0, 1, subgraph_incidence, h, max_nodes_per_hop)
    ##################################

    # 按照one-hot?方法进行编码
    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()     # 余下（子图）的节点对应完整图中节点编号
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)   # enclosing的比例
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)     # 筛掉的节点个数

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    # e = enumerate(sgs_single_root)
    # for i,j in e:
    #     print(i,j)
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    # sg为去掉roots的子图
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes

