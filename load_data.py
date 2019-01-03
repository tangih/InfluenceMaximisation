import os
import numpy as np
import graph_tool.all as gt


def load_fb_subgraph(node_id):
    path = os.path.join('FB_subgraph', 'facebook', '{}.edges'.format(node_id))
    assert os.path.exists(path), "File not found"
    with open(path, 'r') as file:
        V = set()
        E = []
        lines = file.readlines()
        for line in lines:
            line = line[:-1]  # removes the '\n'
            s = line.split(" ")
            i, j = int(s[0]), int(s[1])
            E.append((i, j))
            V.add(i)  # if node is already in V, dismiss
            V.add(j)
        V = list(V)
    return V, E


def compute_weights(V, E):
    n = len(V)
    W = np.zeros((n, n), dtype=np.float)
    for (i, j) in E:
        W[i, j] = np.random.uniform(0, .1)
    return W


def draw_graph(V, E):
    g = gt.Graph()
    vertices = []
    index = {}
    for i in range(len(V)):
        index[V[i]] = i
    for _ in V:
        vertices.append(g.add_vertex())
    for (i, j) in E:
        g.add_edge(vertices[index[i]], vertices[index[j]])
    gt.graph_draw(g)


if __name__ == '__main__':
    V, E = load_fb_subgraph(0)
    # g = price_network(1500)
    # print(g)
    draw_graph(V, E)
