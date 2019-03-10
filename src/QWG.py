from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from pprint import pprint
import numpy as np


class QWG:

    def __init__(self, adjacency_matrix, initial_position, coin, step):
        '''
        adjacency_matrix ... array like (n by n matrix) FIXME 動的にぽちぽち
        initial_position ... integer (point out the number from 0 to len(adjacency_matrix))
        coin ... array like (2 by 2 or ... unitary matrix)
        step ... integer (the number of steps a walker move)
        '''
        self.ajmatrix = adjacency_matrix
        self.initial_position = initial_position
        self.coin = coin
        self.step = step

    def encoding(self):
        pass

if __name__ == '__main__':
    ajmatrix = np.array(((0, 1, 0, 0, 0, 0, 0, 1),
                         (1, 0, 1, 0, 0, 0, 0, 0),
                         (0, 1, 0, 1, 0, 0, 0, 0),
                         (0, 0, 1, 0, 1, 0, 0, 0),
                         (0, 0, 0, 1, 0, 1, 0, 0),
                         (0, 0, 0, 0, 1, 0, 1, 0),
                         (0, 0, 0, 0, 0, 1, 0, 1),
                         (1, 0, 0, 0, 0, 0, 1, 0)))  # Circle TODO expand
    hadamard_coin = 1/np.sqrt(2)*np.array(((1, 1),
                                           (1, -1)))
    test = QWG(ajmatrix, initial_position=0, coin=hadamard_coin, step=1)
    test
