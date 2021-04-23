import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (1, 13), (14, 13), (15, 14),
                (16, 15), (1, 17), (18, 17), (19, 18), (20, 19),
                (22, 8), (23, 8), (24, 12), (25, 12)]
neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]

nodes = [i for i in range(25)]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(neighbor_link)
G = nx.convert_node_labels_to_integers(G, first_label=0)

self_link = [(int(i), int(i)) for i in G]

As = [np.concatenate((np.array(G.edges), self_link), axis=0)]
nodes_l = []
nodes_l.append(nodes)
lvls = 4  # 25 -> 11 -> 5 -> 1


for _ in range(lvls-1):
    stay  = []
    start = 1
    while True:
        remove = []
        for i in G:
            if len(G.edges(i)) == start and i not in stay:
                lost = []
                for j,k in G.edges(i):
                    stay.append(k)
                    lost.append(k)
                recon = [(l,m) for l in lost for m in lost if l!=m]
                G.add_edges_from(recon)            
                remove.append(i)

        if start>10: break  # Remove as maximum as possible
        G.remove_nodes_from(remove)

        cycle = nx.cycle_basis(G)  # Check if there is a cycle in order to downsample it
        if len(cycle)>0:
            if len(cycle[0])==len(G):
                last = [x for x in G if x not in stay]
                G.remove_nodes_from(last)

        start+=1

    

    mapping = {}
    for i, x in enumerate(G): mapping[int(x)] = i; print(i, x)
    G = nx.relabel_nodes(G, mapping)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    nodes = [i for i in range(len(G))]
    nodes_l.append(nodes)

    self_link = [(int(i), int(i)) for i in G]
    G_l = np.concatenate((np.array(G.edges), self_link), axis=0) if len(np.array(G.edges)) > 0 else self_link
    As.append(G_l)
    

    nx.draw(G, with_labels = True)
    plt.show()
    


print(As)
assert len(As)==lvls



