from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from pprint import pprint
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class QWG:

    def __init__(self, adjacency_matrix, initial_position, coin, step):
        '''
        adjacency_matrix ... array like (n by n matrix) FIXME 動的にぽちぽち
        initial_position ... integer (point out the number from 0 to len(adjacency_matrix))
        coin ... string TODO(array like (2 by 2 or ... unitary matrix))
        step ... integer (the number of steps a walker move)
        '''
        self.ajmatrix = adjacency_matrix
        self.initial_position = initial_position
        self.coin = coin
        self.step = step

    def encoding(self):
        '''
        encoder from graph (adjacency matrix) to quantum circuit
        '''
        pass

    def make_graph(self, settings={'img_size': (8, 8),
                                   'node_size': 600, 
                                   'font_size': 13,
                                   'colors': ['#33FF00', '#FFCC00', 'black']}):
        '''
        settings:
            img_size -> tuple (size of image)
            node_size -> integer (size of each node)
            font_size -> integer (size of label font)
            'colors' -> list (['edge color', 'node color', 'label fornt color'])
        '''
        bins = self.twomod(len(self.ajmatrix))
        nodes = [format(i, '0%sb' % (bins)) for i, _ in enumerate(self.ajmatrix)]  # TODO benchmark
        plt.figure(figsize=settings['img_size'])
        G = nx.Graph()
        G.add_nodes_from(nodes)
        edges = []
        for hi, hv in enumerate(self.ajmatrix):
            for wi, wv in enumerate(hv):
                if(wv):
                    edges.append((nodes[hi], nodes[wi]))
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, pos, edge_color=settings['colors'][0])
        nx.draw_networkx_nodes(G, pos, node_color=settings['colors'][1], node_size=settings['node_size'])
        nx.draw_networkx_labels(G, pos, font_size=settings['font_size'], font_color=settings['colors'][2])
        plt.axis("off")
        plt.show()

    @staticmethod
    def twomod(x):
        bin_node = 0
        while True:
            if x == 1:
                break
            elif x % 2 == 0:
                x = x / 2
                bin_node += 1
            else:
                x += 1
        return bin_node

if __name__ == '__main__':
    ajmatrix = np.array(((0, 1, 0, 1, 0, 0, 0, 1),
                         (1, 0, 1, 1, 0, 0, 0, 0),
                         (0, 1, 0, 1, 0, 1, 0, 0),
                         (0, 0, 1, 0, 1, 0, 0, 0),
                         (0, 0, 0, 1, 0, 1, 0, 0),
                         (0, 0, 0, 0, 1, 0, 1, 0),
                         (0, 0, 0, 0, 0, 1, 0, 1),
                         (1, 0, 0, 0, 0, 0, 1, 0)))  # Circle TODO expand
    ajmatrix = np.array(np.random.choice([0, 1], (10, 10)))
    hadamard_coin = 1/np.sqrt(2)*np.array(((1, 1),
                                           (1, -1)))
    test = QWG(ajmatrix, initial_position=0, coin="H", step=1)
    test.make_graph()
