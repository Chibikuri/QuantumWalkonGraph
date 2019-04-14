
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from pprint import pprint
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class RandomWalk:

    def __init__(self, eta, d):
        self.eta = eta
        self.d = d

    def isomorph(self, G1, G2):
        '''
        Input:
            G1, G2 : Matrix(array, N*N)
        Output:
            Bool: if two graph is isomorph, then return True. Otherwise False.
        '''
        if len(G1) != len(G2):
            raise Exception("The number of nodes are different!")
        self.N = len(G1)
        s = 0
        D = []
        P = 0
        # a = self.x_star(G1, D)
        while s < self.N:
            D.append(np.random.random())
            # print(D)
            Z1 = self.Z_star(G1, D)
            Z2 = self.Z_star(G2, D)
            P = self.RebuildPermutation(Z1, Z2)
            if self.isSuccess(G1, G2, P):
                break
            s += 1

    @staticmethod
    def permutation_matrix(n):
        A = [[1 if j == i else 0 for j in range(n)] for i in range(n)]
        PMX = [np.array([row for row in m]).reshape(n, n) for m in itertools.permutations(A)]
        return PMX
    
    def RebuildPermutation(self, Z1, Z2):
        if len(Z1) != len(Z2):
            raise Exception("Matrix size is different!")
        permulist = self.permutation_matrix(len(Z1))
        for pmat in permulist:
            print(np.array(pmat).shape)
            # FIXME How to rebuild permutation matrix?
        #     if np.dot(pmat.T, np.dot(Z1, pmat)) == Z2:
        #         return pmat
        #         break
        # return pmat[0]

    def x_star(self, G, D):
        # print("G", G)
        Adja = G
        Theta = np.diag([1/sum(i) for i in G])
        W = np.dot(Theta, Adja)
        vec = [1 for i in G]
        stlist = [((1 - d)/self.N)*(np.identity(self.N) - d*W)**(-1)*vec for d in D]
        # print("listx", stlist)
        return stlist
    
    def y_star(self, G, D):
        Adja = G.T
        Theta = np.diag([1/sum(i) for i in G.T])
        W = np.dot(Theta, Adja)
        # vec = [1 for i in G]
        print("vec", vec)
        stlist = [((1 - d)/self.N)*(np.identity(self.N) - d*W)**(-1)*vec for d in D]
        # print("listy", stlist)
        return stlist

    def Z_star(self, G, D):
        Z = []
        # for i in range(G.shape[0]):
        for i, j in zip(self.x_star(G, D)[0], self.y_star(G, D)[0]):
            # print([i.tolist() + j.tolist()])
            Z.append(i.tolist() + j.tolist())
        # pprint(self.x_star(G, D)[0].tolist())
        print(np.array(Z).shape)
        return Z

    def isSuccess(self, A1, A2, P):
        '''
        A1, A2 is adjacency matrix
        P is the permutation matrix.
        if P'A1P = A2 return True other wise return False
        '''
        return True

    def visualize(self, adjacency):
        nodes = np.array(['a', 'b', 'c', 'd'])
        G = nx.Graph()
        G.add_nodes_from(nodes)
        edges = []
        for hi, hv in enumerate(adjacency):
            for wi, wv in enumerate(hv):
                if(wv):
                    edges.append((nodes[hi], nodes[wi]))
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, with_labels=True)
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    test = RandomWalk(0.2, 0.2)
    G1 = np.array(((0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0)))
    G2 = np.array(((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)))
    test.isomorph(G1, G2)
    test.permutation_matrix(3)