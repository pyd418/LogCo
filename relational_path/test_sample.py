import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
from scipy.sparse import coo_matrix
import sys
import torch
import torch.nn as nn
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import ssp_multigraph_to_dgl
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx
import itertools
import dgl
import copy

# a = np.random.uniform()
# # np.random.choice()
# print(a)

# row = np.array([0,0,2,2,3,5,6]) # 行索引
# col = np.array([1,2,3,5,1,3,4]) # 列索引
# data = np.array([1,1,1,1,1,1,1]) # 索引对应的数值
# csc = ssp.csc_matrix((data, (row, col)), shape=(7, 7))
# # csc_1.tocoo()

# row_1 = np.array([10,10,12,15,16]) # 行索引
# col_1 = np.array([11,12,13,13,14]) # 列索引
# data_1 = np.array([1,1,1,1,1]) # 索引对应的数值
# csc_1 = ssp.csc_matrix((data_1, (row_1, col_1)), shape=(17, 17))
#
# row_2 = np.array([12,12,13]) # 行索引
# col_2 = np.array([13,15,11]) # 列索引
# data_2 = np.array([1,1,1]) # 索引对应的数值
# csc_2 = ssp.csc_matrix((data_2, (row_2, col_2)), shape=(17, 17))

row_1 = np.array([10,13,10]) # 行索引
col_1 = np.array([11,11,13]) # 列索引
data_1 = np.array([1,1,1]) # 索引对应的数值
csc_1 = ssp.csc_matrix((data_1, (row_1, col_1)), shape=(17, 17))

row_2 = np.array([10,12,13]) # 行索引
col_2 = np.array([12,11,12]) # 列索引
data_2 = np.array([1,1,1]) # 索引对应的数值
csc_2 = ssp.csc_matrix((data_2, (row_2, col_2)), shape=(17, 17))

row_3 = np.array([10]) # 行索引
col_3 = np.array([11]) # 列索引
data_3 = np.array([1]) # 索引对应的数值
csc_3 = ssp.csc_matrix((data_3, (row_3, col_3)), shape=(17, 17))

row_4 = np.array([12,12,13]) # 行索引
col_4 = np.array([13,15,11]) # 列索引
data_4 = np.array([1,1,1]) # 索引对应的数值
csc_4 = ssp.csc_matrix((data_4, (row_4, col_4)), shape=(17, 17))


def get_neighbor_nodes(roots, adj, h=5, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def findAllPath(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]

    paths = []  # 存储所有路径
    for node in graph[start]:
        if node not in path:
            newpaths = findAllPath(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_paths(roots, end, adj, h, max_nodes_per_hop=None, path=[]):
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
                    newpaths = find_paths(node, end, adj, h - 1, max_nodes_per_hop, path)

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


# def obtain_rel_paths(rel_path):
    # all_rel_paths = []
    # alter_rel=[]
    # path_len = len(rel_path)
    # for i in range(path_len):
    #     alter_rel = rel_path[i]
    #     alter_len = len(rel_path[i])
    #     # all_rel_paths = rel_path[i]
    #     for more_rel in range(alter_len):
    #         print(rel_path[i][more_rel])
    #         alter_rel.append(rel_path[i][more_rel])


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


def main():
    csc_list = [csc_1, csc_2, csc_3, csc_4]

    rel_num = len(csc_list)
    csc = incidence_matrix(csc_list)
    # print(csc_1.tocoo().row)
    csc_dgl = ssp_multigraph_to_dgl(csc_list)
    csc_dgl_new = dgl.DGLGraph(csc_dgl)
    nodes = [10, 11, 12, 13, 14, 15, 16]
    nodes_2 = [10, 11, 12, 13]
    h = 5
    subgraph = csc_dgl_new.subgraph(nodes)
    subgraph_2 = csc_dgl_new.subgraph(nodes_2)
    # subgraph_3 = ssp_multigraph_to_dgl([csc_3, csc_4])

    edge = subgraph.edge_id(2, 3)
    # print(subgraph.edge_id(5, 3))
    subgraph.edata['type'] = csc_dgl.edata['type'][csc_dgl.subgraph(nodes).parent_eid]
    subgraph_2.edata['type'] = csc_dgl.edata['type'][csc_dgl.subgraph(nodes_2).parent_eid]

    n1_nodes = subgraph.number_of_nodes()
    n1_ids = list(range(n1_nodes))
    n2_nodes = subgraph_2.number_of_nodes()
    n2_ids = list(range(n2_nodes))

    subgraph.ndata['id'] = np.array(n1_ids)
    subgraph_2.ndata['id'] = np.array(n2_ids)
    print(subgraph.nodes[6].data['id'])
    print(subgraph_2.nodes[1].data['id'])
    print(subgraph.edata['type'][edge])

    subgraph_list = [subgraph, subgraph_2]
    batched_graph = dgl.batch(subgraph_list)
    # print(batched_graph.edge_id(5, 3))
    # print(subgraph.parent_eid)
    # print(csc_dgl.successors(0))


# ##########  dgl_subgraph ###################################################
    print("dgl process: ")
    dgl_paths = obtain_dglpaths(subgraph, 0, 1, 5)
    print(dgl_paths)
    dgl_relation_exist(subgraph, 0, 1)
    all_rel_paths = obtain_dgl_relation_paths(dgl_paths, subgraph)
    # all_rel_paths = obtain_dgl_relation_paths(dgl_paths, subgraph_3)
    print(all_rel_paths)
    # rel_path_list = []
    rel_path_list = unfold_rel_paths(all_rel_paths)
    print(rel_path_list)
    print(create_neg_paths(rel_path_list, 0, rel_num))
    # if [0,0,1] in rel_path_list:
    #     print('True')


# ##########  batch_subgraph #################################################
    print("dgl batch process: ")
    g = list(dgl.unbatch(batched_graph))

    print(g[0].nodes[0, 1, 2, 3, 4, 5, 6].data['id'])
    print(g[1].nodes[0, 1].data['id'])

    root_list = np.zeros(len(g), dtype=int)
    end_list = np.ones(len(g), dtype=int)
    hop_list = np.ones(len(g), dtype=int) * h

    g_new = dgl.DGLGraph(g[0])
    print(g_new.edge_ids(2, 3))
    # print(g_new.edata['type'][g_edge].tolist())

    param_1 = zip(g, root_list, end_list, hop_list)
    dgl_paths_batch = list(itertools.starmap(obtain_dglpaths, list(param_1)))
    print(dgl_paths_batch)

    param_2 = zip(dgl_paths_batch, g)
    dgl_relpath_batch = list(itertools.starmap(obtain_batch_dgl_rel_paths, list(param_2)))
    print(dgl_relpath_batch)

    param_3 = dgl_relpath_batch
    rel_path_batch = list(map(unfold_rel_paths, param_3))
    print(rel_path_batch)

    r_labels = np.zeros(len(g), dtype=int)
    rel_list = np.ones(len(g), dtype=int) * rel_num
    param_4 = zip(rel_path_batch, r_labels, rel_list)
    neg_paths_batch = list(itertools.starmap(create_neg_paths, list(param_4)))
    print(neg_paths_batch)

    # print(batched_graph.edata['type'])
    # print(np.random.choice([0,1]))

# ###########################################################################
#     allpath = find_paths(0, 1, csc, 5)
#     find_relations(subgraph, allpath[0][0], allpath[0][1])
#     all_rel_paths = find_relation_paths(allpath, 0, 1, 0, csc_list)
#     # print(obtain_rel_paths(all_rel_paths[1]))
#     rel_path_list = []
#     for rel_paths in all_rel_paths:
#         rel_path = obtain_rel_paths(rel_paths)
#         rel_path_list=rel_path_list + rel_path
#     print(rel_path_list)
#     print(allpath)
#     print(all_rel_paths)
# ###########################################################################

###########################################################################
    m = nn.Softmax(dim=0)  # 注意是沿着哪个维度计算
    # input = torch.randn(1, 3)
    a = [36, 2]
    # print(torch.exp(torch.Tensor([1000])))
    input= torch.Tensor(a)
    print("input:")
    print(input)
    output = m(input)
    print("output:")
    print(output)

###########################################################################
    rel_emb = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    rel_tensor = []
    path_1 = [0, 1, 0]
    path_2 = [1, 0, 1]
    # define list
    # array = np.asarray(path)
    # the second method
    array = np.array(path_1, dtype=int)
    rel_emb = torch.Tensor(rel_emb)
    print(rel_emb[path_1])
    a = torch.mean(rel_emb[array], dim=0).tolist()
    print(a)

    # rel_tensor.append(a)
    # # rel_tensor = torch.cat((rel_tensor, rel_emb), dim = 0)
    # print(rel_tensor)

###########################################################################
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    paths = [[[1, 2, 3], [4, 5, 6]], [[0, 1, 2]], [[4, 5, 6]]]
    neg_paths = [[[0, 1, 2], [1, 2, 3]], [[1, 2, 3]], [[0, 1, 2]]]
    rel = torch.Tensor([1, 1])
    a = torch.Tensor([0.2, 0.2, 0.6])
    b = torch.Tensor([[2, 1], [3, 5], [7, 8]])
    c = torch.Tensor([0.1, 0.2, 0.3])
    y = torch.Tensor([[0, 1, 2], [1, 2, 3], [4, 5, 6], [6, 7, 8]])
    z = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    print("new output")
    print(torch.matmul(a, b))
    # print(torch.sum(torch.mul(x, z), dim=1))

    for path in zip(paths, neg_paths):
        # print(torch.matmul(torch.Tensor(path), rel))
        print(path[0], path[1])

# ###########################################################################
#     print(torch.exp(a))
#     print(-(a + a))
#     print(torch.sum(b, dim=0))
    print(rel_emb[path_1].shape)
    print(rel_emb[path_2].shape)
    print(torch.cat((rel_emb[path_1], rel_emb[path_2]), dim=1))
#     sim = torch.cat((a.view(1, len(a)), c.view(1, len(c))), dim=0)
#     print(sim)
#     print(m(sim)[0])
#     print(a)
#     zero_id = (a > 0.1).nonzero(as_tuple=False).flatten()
#     a[zero_id] += 1e-4
#     print(torch.log(torch.Tensor([0]) + 1e-10))
#     # print((a > 0.1).nonzero(as_tuple=False).flatten())
# ###########################################################################

# ###########################################################################
#     criterion = nn.CrossEntropyLoss()
#     output = torch.Tensor([0.1, 0.2, -0.3, 0.4, -0.5])
#     label = torch.empty(1, dtype=torch.long).random_(5)
#     loss = criterion(output.view(1, 5), label)
#
#     print("output:")
#     print(output.view(1, len(output)))
#     print("label:")
#     print(label)
#     print("loss:")
#     print(loss)
# ###########################################################################
def main_1():
    # rnn = nn.GRU(10, 10, 2)
    # x = torch.randn(3, 1, 10)
    # print(x)
    # out, ht = rnn(x)
    # print(out)
    # print(ht)
    conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2)
    input = torch.randn(3, 3, 32)
    input = input.permute(0, 2, 1)
    output = conv1(input)
    print(input)
    print(output)


if __name__ == '__main__':
    # main()
    main_1()
