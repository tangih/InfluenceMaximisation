import os
import numpy as np
#import graph_tool.all as gt


def load_graph(network, node_id):
    """
    loads subgraph from SNAP dataset (https://snap.stanford.edu/data/)
    :param network: name of the graph
    :param node_id: id of the node to load
    :return: E the list of edges, W the associated weight (ramdomly picked) and n s.t. V = \{1, ..., n\}
    """
    path = os.path.join(network, '{}.edges'.format(node_id))
    assert os.path.exists(path), "File not found"
    with open(path, 'r') as file:
        V = {}
        E = []
        W = []
        cnt = 0
        lines = file.readlines()
        for line in lines:
            s = line[:-1].split(' ')
            i, j = int(s[0]), int(s[1])
            if i not in V.keys():
                V[i] = cnt
                cnt += 1
            if j not in V.keys():
                V[j] = cnt
                cnt += 1
            E.append((V[i], V[j]))
            W.append(np.random.uniform(0, .1))
    return E, W, cnt


def draw_graph(E, W, n):
    """
    Draws the graph, using the same graph format as in load_graph
    """
    g = gt.Graph()
    vertices = []
    for _ in range(n):
        vertices.append(g.add_vertex())
    for (i, j) in E:
        g.add_edge(vertices[i], vertices[j])
    gt.graph_draw(g)


if __name__ == '__main__':
    E, W, n = load_graph('twitter', 12831)
    # g = price_network(1500)
    # print(g)
    draw_graph(E, W, n)
