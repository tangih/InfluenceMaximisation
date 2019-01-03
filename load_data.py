import numpy as np
import os


def FB_subgraph(path):
    
    assert os.path.exists(path + "/facebook_combined.txt"), "File not found"
    file = open(path + "/facebook_combined.txt", "r")
    V = set()
    E = []
    edges = file.readlines()
    s = edges[0].split(" ")
    for e in edges:
        s = e.split(" ")
        i, j = int(s[0]), int(s[1].split("\n")[0])
        E.append((i, j))
        V.add(i)
        V.add(j)
    V = list(V)
    n = len(V)
    W = np.zeros([n, n])
    for (i, j) in E:
        W[i, j] = np.random.uniform(low=0, high=0.1)
        
    return W
        