import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
from scipy.sparse import coo_matrix
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx
import itertools
import copy


def get_neighbor_nodes(roots, adj, h=5, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def find_entity_paths(roots, end, adj, h, max_nodes_per_hop=None, path=[]):
    roots = set([roots])
    path = path + list(roots)
    if roots == set([end]):
        return [path]
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    nodeList = next(bfs_generator)
    paths = list()
    for node in nodeList:
        try:
            # nodeList=next(bfs_generator)
            if node not in path:
                if h == 0:
                    break
                else:
                    newpaths = find_entity_paths(node, end, adj, h - 1, max_nodes_per_hop, path)

                    for newpath in newpaths:
                        paths.append(newpath)
        except StopIteration:
            pass
    return paths


def find_relations(matrix, head, tail):
    if matrix[head, tail] > 0:
        print("Existed")


def find_relation_paths(paths, root, end, rel, A):
    rel_paths = list()
    for path in paths:
        rel_path = list()
        for i in range(len(path) - 1):
            path_rel_label = []
            for adj in range(len(A)):
                # rel_ = adj
                exist_rel = A[adj][path[i], path[i + 1]]
                if exist_rel > 0:
                    path_rel_label.append(adj)
                    pre_path = rel_path
            rel_path = pre_path + [list(path_rel_label)]
        rel_paths.append(rel_path)
    return rel_paths


def obtain_rel_paths_old(rel_path):
    all_list = list(itertools.product(*rel_path))
    return all_list


def find_paths(roots, end, adj, h, max_nodes_per_hop=None, path=[]):
    roots=set([roots])
    path = path + list(roots)
    if roots == set([end]):
        return [path]
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    nodeList = next(bfs_generator)
    paths = list()
    for node in nodeList:
        try:
            # nodeList=next(bfs_generator)
            if node not in path:
                if h == 0:
                    break
                else:
                    newpaths = find_paths(node, end, adj, h - 1, max_nodes_per_hop, path)

                    for newpath in newpaths:
                        paths.append(newpath)
        except StopIteration:
            pass
    return paths


##########################################################################
#dgl graph
##########################################################################

def obtain_dglpaths(graph, root, end, h, path=[]):
    roots = set([root])
    path = path + list(roots)
    if roots == set([end]):
        return [path]
    nodeList = list(graph.successors(root))
    paths = list()
    for node in nodeList:
        try:
            # nodeList=next(bfs_generator)
            node = int(node)
            if node not in path:
                if h == 0:
                    break
                else:
                    newpaths = obtain_dglpaths(graph, node, end, h - 1, path)

                    for newpath in newpaths:
                        paths.append(newpath)
        except StopIteration:
            pass
    return paths


def dgl_relation_exist(dgl_graph, head, tail):
    if dgl_graph.edge_id(head, tail) is not None:
        print("Existed")


def obtain_dgl_relation_paths(paths, dgl_graph):
    rel_paths = list()
    pre_path = []
    for path in paths:
        rel_path = list()
        for i in range(len(path) - 1):
            path_rel_label = []
            if dgl_graph.edge_id(path[i], path[i + 1]) is not None:
                rel_label = dgl_graph.edata['type'][dgl_graph.edge_id(path[i], path[i + 1])].tolist()
                path_rel_label.append(rel_label)
                pre_path = rel_path
            rel_path = pre_path + path_rel_label
        rel_paths.append(rel_path)
    return rel_paths


def obtain_batch_dgl_rel_paths(paths, dgl_graph):
    rel_paths = list()
    pre_path = []
    for path in paths:
        rel_path = list()
        for i in range(len(path) - 1):
            path_rel_label = []
            if dgl_graph.edge_id(path[i], path[i + 1]) is not None:
                rel_label = dgl_graph.edata['type'][dgl_graph.edge_ids(path[i], path[i + 1])].tolist()
                path_rel_label.append(rel_label)
                pre_path = rel_path
            rel_path = pre_path + path_rel_label
        rel_paths.append(rel_path)
    return rel_paths


def obtain_rel_path(rel_path):
    all_list = list(itertools.product(*rel_path))
    tuple_list=[]
    for tuples in all_list:
        tuple_list.append(list(tuples))
    return tuple_list


def unfold_rel_paths(folded_paths):
    unfolded_paths = []
    for rel_paths in folded_paths:
        rel_path = obtain_rel_path(rel_paths)
        unfolded_paths = unfolded_paths + rel_path
    return unfolded_paths


##########################################################################
# neg_rel
##########################################################################
def create_neg_paths(paths, target_rel, rels):
    target_list = [target_rel]
    neg_path = []
    old_path = copy.deepcopy(paths)
    # set_old_path = set([old_path])
    for item in paths:
        loop = 0
        while True and loop < rels:
            rel_list = list(range(rels))
            loop += 1
            if item != target_list:
                neg_position = np.random.choice(len(item))
                rel_list.remove(item[neg_position])
                item_temp = copy.deepcopy(item)
                item_temp[neg_position] = np.random.choice(rel_list)
                if item_temp not in old_path:
                    neg_path.append(item_temp)
                    break
                # item[neg_position] = np.random.choice(rel_list)
                # if item not in old_path:
                #     neg_path.append(item)
                #     break
        else:
            neg_list = list(range(rels))
            neg_list.remove(target_rel)
            item = [np.random.choice(neg_list)]
            neg_path.append(item)
    return neg_path

def create_mul_neg_paths(paths, target_rel, rels):
    target_list = [target_rel]
    neg_path = []
    old_path = copy.deepcopy(paths)
    # set_old_path = set([old_path])
    for item in paths:
        loop = 0
        while True and loop < rels:
            rel_list = list(range(rels))
            loop += 1
            if item != target_list:
                neg_position = np.random.choice(len(item))
                rel_list.remove(item[neg_position])
                item_temp = copy.deepcopy(item)
                item_temp[neg_position] = np.random.choice(rel_list)
                if item_temp not in old_path:
                    neg_path.append(item_temp)
                    break
                # item[neg_position] = np.random.choice(rel_list)
                # if item not in old_path:
                #     neg_path.append(item)
                #     break
        else:
            neg_list = list(range(rels))
            neg_list.remove(target_rel)
            item = [np.random.choice(neg_list)]
            neg_path.append(item)
    return neg_path
