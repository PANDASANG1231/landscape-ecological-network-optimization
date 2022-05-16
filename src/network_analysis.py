import os
import numpy as np
import pandas as pd
import networkx as nx

def generate_graph(node_list, edges_list):    
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edges_list)
    return G

def calculate_pij(G):
    return nx.adjacency_matrix(G).todense()

def calculate_distance(nodes):
    m = nodes.shape[0]
    distance = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            cord1, cord2 = nodes.loc[i, ["X坐标", "Y坐标"]].values, nodes.loc[j, ["X坐标", "Y坐标"]].values
            dist = ((cord2 - cord1)**2).sum() ** 0.5
            distance[i, j] = dist
            distance[j, i] = dist
    one_div_one_plus_dk = 1 / (distance + 1)
    return one_div_one_plus_dk

def calculate_IIPC(G, one_div_one_plus_dk, E_list):
    pij = calculate_pij(G)
    A = np.multiply(pij, one_div_one_plus_dk)
    IIPC = np.dot(E_list.reshape(1, -1), np.dot(A, E_list.reshape(-1, 1)))
    return IIPC[0, 0]

def calculate_ITSI(G, i, j, one_div_one_plus_dk, E_list):
    G1 = G.copy()
    G1.add_edges_from([(i, j)])
    return calculate_IIPC(G1, one_div_one_plus_dk, E_list) / calculate_IIPC(G, one_div_one_plus_dk, E_list) - 1