from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from pprint import pprint
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import time


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
        self.N = len(adjacency_matrix)
        self.n = math.ceil(math.log(self.N))

    def encoding(self):
        '''
        encoder from graph (adjacency matrix) to quantum circuit
        This version cost too much resource. FIXME
        '''
        graph_x = QuantumRegister(self.n, 'graph_x')
        graph_y = QuantumRegister(self.n, 'graph_y')
        Adjacency_xy = QuantumRegister(1, 'ajxy')
        state_x = QuantumRegister(self.n, 'statex')
        state_y = QuantumRegister(self.n, 'statey')
        ancilla = QuantumRegister(4*self.n, 'anc')
        classicala = ClassicalRegister(self.n, 'result1')
        classicalb = ClassicalRegister(self.n, 'result2')
        
        backend = Aer.get_backend('qasm_simulator')
        # print(Qwalk)
        #  ---experience---
        count_results = []
        start = time.time()
        # HACK more efficient way
        for t in range(self.step):
            for x, _ in enumerate(self.ajmatrix):
                for y, Axy in enumerate(self.ajmatrix[x]):
                    Qwalk = QuantumCircuit(graph_x, graph_y, Adjacency_xy, state_x, state_y, ancilla, classicala, classicalb)
                    print(self.ajmatrix[x])
                    print("initial params!", x, y, Axy)
                    self.initialize(Qwalk, graph_x, graph_y, state_x, state_y, x, y, Axy)
                    self._compute(Qwalk, graph_x, graph_y, Adjacency_xy, state_x, state_y, ancilla)
                    Qwalk.measure(state_x, classicala)
                    Qwalk.measure(state_y, classicalb)
                    job = execute(Qwalk, backend=backend, shots=1024)
                    result = job.result()
                    count = result.get_counts(Qwalk)
                    count_results.append(count)
                    print(Qwalk)
        print(count_results)
        end = time.time() - start
        print(end)
        

    def _compute(self, qc, graphx, graphy, Axy, statex, statey, ancilla):
        xglist = [i for i in graphx]
        xslist = [i for i in statex]
        xlist = xglist + xslist
        xlist.append(ancilla[0])

    def initialize(self, qc, graphx, graphy, statex, statey, x, y, Axy):
        '''
        qc -> Quantum Circuit(object)
        graph -> Quantum Register of graph part(object)
        state -> Quantum Register of state part(object)
        x -> integer(x col of ajmatrix)
        y -> integer(y row of ajmatrix)
        Axy -> integer(value of ajmatrix)
        '''
        graph_x = list(format(x, '0%sb' % (self.n)))
        graph_y = list(format(y, '0%sb' % (self.n)))
        for index, ini_x in enumerate(graph_x):
            print("index", ini_x=="1", ini_x)
            if ini_x == '1':
                qc.x(graphx[index])
                qc.x(statex[index])

        for index, ini_y in enumerate(graph_y):
            print("index", index)
            if ini_y == '1':
                qc.x(graphy[index])
                qc.x(statey[index])

        return qc

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

    def _cnx(self, qc, *qubits):
        if len(qubits) == 3:
            qc.ccx(*qubits)
        elif len(qubits) > 3:
            last = qubits[-1]
            qc.crz(np.pi/2, qubits[-2], qubits[-1])
            qc.cu3(np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.cu3(-np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.crz(-np.pi/2, qubits[-2], qubits[-1])
        elif len(qubits) == 2:
            qc.cx(*qubits)
    
    def _cnwx(self, qc, *qubits):
        for i in qubits[self.subnodes:-1]:
            print(i)
            qc.x(i)
        if len(qubits) == 3:
            qc.ccx(*qubits)
        elif len(qubits) > 3:
            last = qubits[-1]
            qc.crz(np.pi/2, qubits[-2], qubits[-1])
            qc.cu3(np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.cu3(-np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.crz(-np.pi/2, qubits[-2], qubits[-1])
        elif len(qubits) == 2:
            qc.cx(*qubits)
        for j in qubits[self.subnodes:-1]:
            print("jj", j)
            qc.x(j)

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
    ajmatrix = np.array(((0, 1, 0, 0, 0, 0, 0, 1),
                         (1, 0, 1, 0, 0, 0, 0, 0),
                         (0, 1, 0, 1, 0, 0, 0, 0),
                         (0, 0, 1, 0, 1, 0, 0, 0),
                         (0, 0, 0, 1, 0, 1, 0, 0),
                         (0, 0, 0, 0, 1, 0, 1, 0),
                         (0, 0, 0, 0, 0, 1, 0, 1),
                         (1, 0, 0, 0, 0, 0, 1, 0)))  # Circle TODO expand
    ajmatrix = np.array(((0, 1),
                         (1, 0)))
    # ajmatrix = np.array(np.random.choice([0, 1], (10, 10)))
    hadamard_coin = 1/np.sqrt(2)*np.array(((1, 1),
                                           (1, -1)))
    test = QWG(ajmatrix, initial_position=0, coin="H", step=1)
    # test.make_graph()
    test.encoding()
