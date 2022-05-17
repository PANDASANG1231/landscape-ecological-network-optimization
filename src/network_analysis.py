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


def convert_matrix_to_rank(df_rst, head=0, end=20):
    df_rk = []
    for i in df_rst.index:
        for j in df_rst.columns:
            if j > i:
                df_rk.append([i, j, df_rst.loc[i, j]])
                
    df_rk = pd.DataFrame(df_rk, columns=["start", "end", "value"])       
    df_rk.sort_values("value", ascending=False, inplace=True)
    df_rk = df_rk.reset_index(drop=True).reset_index(drop=False).rename({"index":"rank"}, axis=1)
    return df_rk.head(end).tail(end-head)

def gen_new_graph(df_rk):
    
    edges_list_new = []
    for idx, i in df_rk.iterrows():
        edges_list_new.append([int(i["start"]), int(i["end"])])

    G_new = nx.Graph()
    G_new.add_edges_from(edges_list_new)
    return G_new


def plot_network_graph(G, nodes):
    
    import plotly.graph_objects as go

    edge_x = []
    edge_y = []
    for edge in G.edges():

        x0, y0 = nodes.loc[nodes["斑块编号"] == edge[0], ["X坐标", "Y坐标"]].values[0]
        x1, y1 = nodes.loc[nodes["斑块编号"] == edge[1], ["X坐标", "Y坐标"]].values[0]

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    node_x = []
    node_y = []
    node_t = []
    
    for node in G.nodes():
        x, y = nodes.loc[nodes["斑块编号"] == node, ["X坐标", "Y坐标"]].values[0]
        node_x.append(x)
        node_y.append(y)
        node_t.append(node)
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')


    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_t,
        mode='markers+text',
        hoverinfo='text',
        textposition="middle center",
        textfont=dict(
            size=10,
            color="White"
        ),
        marker=dict(
            showscale=False,
            colorscale='Viridis',
            reversescale=True,
            color="#5D69B1",
            size=20,
            # colorbar=dict(
            #     thickness=15,
            #     title='Node Connections',
            #     xanchor='left',
            #     titleside='right'
            # ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Lanscape Network Visualization',
                        titlefont_size=25,
                        showlegend=False,
                        width=1000,
                        height=600,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )    
    return fig


def add_graph(fig, G_new, nodes, color='firebrick'):
    
    import plotly.graph_objects as go

    edge_x = []
    edge_y = []
    for edge in G_new.edges():

        x0, y0 = nodes.loc[nodes["斑块编号"] == edge[0], ["X坐标", "Y坐标"]].values[0]
        x1, y1 = nodes.loc[nodes["斑块编号"] == edge[1], ["X坐标", "Y坐标"]].values[0]

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.8, color=color, dash='dot'),
        hoverinfo='none',
        mode='lines')

    fig.add_trace(
        edge_trace
    )
    
    return fig