import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class graph_ntu():

    def __init__(self,
                 max_hop=1,
                 dilation=1):
        self.max_hop  = max_hop
        self.dilation = dilation
        self.lvls     = 4  # 25 -> 11 -> 5 -> 1
        self.As       = []
        self.hop_dis  = []

        self.get_edge()
        for lvl in range(self.lvls):
            self.hop_dis.append(get_hop_distance(self.num_node, self.edge, lvl, max_hop=max_hop))
            self.get_adjacency(lvl)

        self.mapping = upsample_mapping(self.map, self.nodes, self.edge, self.lvls)[::-1]

    def __str__(self):
        return self.As

    def get_edge(self):
        self.num_node = []
        self.nodes = []
        self.center = [21 - 1]
        self.nodes = []
        self.Gs = []
        
        neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                        (11, 10), (12, 11), (1, 13), (14, 13), (15, 14),
                        (16, 15), (1, 17), (18, 17), (19, 18), (20, 19),
                        (22, 8), (23, 8), (24, 12), (25, 12)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]

        nodes = np.array([i for i in range(25)])
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(neighbor_link)
        G = nx.convert_node_labels_to_integers(G, first_label=0)

        self_link = [(int(i), int(i)) for i in G]


        self.map = [np.array([[i, x] for i,x in enumerate(G)])]
        self.edge = [np.concatenate((np.array(G.edges), self_link), axis=0)]
        self.nodes.append(nodes)
        self.num_node.append(len(G))
        self.Gs.append(G.copy())


        for _ in range(self.lvls-1):
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

            map_i = np.array([[i, x] for i,x in enumerate(G)])  # Keep track graph indices
            self.map.append(map_i)

            mapping = {}  # Change mapping labels
            for i, x in enumerate(G): 
                mapping[int(x)] = i
                if int(x)==self.center[-1]:
                    self.center.append(i)
            

            G = nx.relabel_nodes(G, mapping)  # Change labels
            G = nx.convert_node_labels_to_integers(G, first_label=0)
            
            nodes = np.array([i for i in range(len(G))])
            self.nodes.append(nodes)

            self_link = [(int(i), int(i)) for i in G]
            G_l = np.concatenate((np.array(G.edges), self_link), axis=0) if len(np.array(G.edges)) > 0 else self_link
            self.edge.append(G_l)
            self.num_node.append(len(G))
            self.Gs.append(G.copy())

        
        '''for i, G in enumerate(self.Gs):  # Uncomment this to visualize graphs
            plt.clf()  # Uncomment this to visualize graphs
            nx.draw(G, with_labels = True)
            plt.savefig('G_' + str(i) + '.pdf')'''

        assert len(self.num_node) == self.lvls
        assert len(self.nodes)    == self.lvls
        assert len(self.edge)     == self.lvls
        assert len(self.center)   == self.lvls
        assert len(self.map)      == self.lvls
        
        
    def get_adjacency(self, lvl):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node[lvl], self.num_node[lvl]))
        for hop in valid_hop:
            adjacency[self.hop_dis[lvl] == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node[lvl], self.num_node[lvl]))
            a_close = np.zeros((self.num_node[lvl], self.num_node[lvl]))
            a_further = np.zeros((self.num_node[lvl], self.num_node[lvl]))
            for i in range(self.num_node[lvl]):
                for j in range(self.num_node[lvl]):
                    if self.hop_dis[lvl][j, i] == hop:
                        if self.hop_dis[lvl][j, self.center[lvl]] == self.hop_dis[lvl][i, self.center[lvl]]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[lvl][j, self.center[lvl]] > self.hop_dis[lvl][i, self.center[lvl]]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        A = np.stack(A)
        self.As.append(A)
            


def get_hop_distance(num_node, edge, lvl, max_hop=1):
    A = np.zeros((num_node[lvl], num_node[lvl]))
    for i, j in edge[lvl]:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node[lvl], num_node[lvl])) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

def upsample_mapping(mapping, nodes, edges, lvls):

    all_hoods = []
    
    i = lvls - 1
    while i > 0:
        n = i - 1

        neighbors = []

        for node in nodes[n]:
            if node not in mapping[i][:,1]:
                hood = []
                for cmap in mapping[i]:
                    hood.append(cmap[0]) if ([node, cmap[1]] in edges[n].tolist()) or ([cmap[1], node] in edges[n].tolist()) else None
                
                if len(hood)>0: hood.insert(0, node)

                if len(hood)>0: neighbors.append(np.array(hood)) 

        all_hoods.append(neighbors)
        
        i-=1

    return all_hoods
