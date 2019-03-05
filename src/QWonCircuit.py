from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute, Aer
import numpy as np

class QWonCircle:
    """
    This is QuantumWalk algorithm on 2d hyper circle.
    """

    def __init__(self, node, order, step):
        """
        node -> Integer : return 2**n
        order -> Integer : The number of edge from one node (2**node-1)
        step -> Integer : How many steps a walker move.
        """
        # if order > node:
        #     raise Exception("Order must be lass than the number of nodes")

        self.nodes = 2**node
        self.cnodes = node
        self.order = order
        self.step = step
        subnode = 0
        while True:
            if order == 1:
                break
            elif order % 2 == 0:
                order = order / 2
                subnode += 1
            else:
                raise Exception("order must be n-th power of 2")
        print(subnode)
        self.n_qubit = node
        self.subnodes = subnode

    def QW(self):
        q = QuantumRegister(self.n_qubit, "node")
        q_subnode = QuantumRegister(self.subnodes, "subn")
        c = ClassicalRegister(self.cnodes)
        qc = QuantumCircuit(q, c, q_subnode)

        self.coin(qc, q_subnode)
        self._incr(qc, q, q_subnode, self.subnodes)
        self._decr(qc, q, q_subnode, self.subnodes)
        # self._cnwx(qc, q_subnode[0], q[0], q[1], q[2])
        qc.measure(q, c)
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend=backend, shots=1024)
        result = job.result()
        count = result.get_counts(qc)
        # print(qc)
        return count

    def coin(self, qc, subnode):
        for i in subnode:
            qc.h(i)
        return qc
    
    def step(self, qc, subnode, block):
        self._incr(qc, node, subnode, block)
        self._decr(qc, node, subnode, block)
        return qc
        
    def _incr(self, qc, node, subnode, block):
        whitenot = [format(i, "0%sb" % (self.n_qubit-2)) for i in range(2**self.subnodes)][0::2]
        control = [i for i in subnode]
        target = [i for i in reversed(node)]
        interval = round(self.n_qubit/block)
        for i, white in zip(range(0, self.n_qubit, interval), whitenot):
            for index, x_gate in enumerate(list(white)):
                if x_gate == "1":
                    qc.x(subnode[index])
            for t in range(interval, 0, -1):
                operation = control + target[i:i+t]
                print(operation)
                self._cnx(qc, *operation)
            for af_index, af_x_gate in enumerate(list(white)):
                if af_x_gate == "1":
                    qc.x(subnode[af_index])
        qc.barrier()

        return qc

    def _decr(self, qc, node, subnode, block):
        whitenot = [format(i, "0%sb" % (self.n_qubit-2)) for i in range(2**self.subnodes)][1::2]
        print("decr", whitenot)
        control = [i for i in subnode]
        target = [i for i in reversed(node)]
        interval = round(self.n_qubit/block)
        for i, white in zip(range(0, self.n_qubit, interval), whitenot):
            for index, x_gate in enumerate(list(white)):
                if x_gate == "1":
                    qc.x(subnode[index])
            for t in range(interval, 0, -1):
                operation = control + target[i:i+t]
                self._cnwx(qc, *operation)
            for af_index, af_x_gate in enumerate(list(white)):
                if af_x_gate == "1":
                    qc.x(subnode[af_index])
        qc.barrier()

        return qc

    def _cnx(self, qc, *qubits):
        if len(qubits) >= 3:
            last = qubits[-1]
            qc.crz(np.pi/2, qubits[-2], qubits[-1])
            qc.cu3(np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.cu3(-np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.crz(-np.pi/2, qubits[-2], qubits[-1])
        elif len(qubits) == 3:
            qc.ccx(*qubits)
        elif len(qubits) == 2:
            qc.cx(*qubits)
    
    def _cnwx(self, qc, *qubits):
        for i in qubits[self.subnodes:-1]:
            qc.x(i)
        if len(qubits) >= 3:
            last = qubits[-1]
            qc.crz(np.pi/2, qubits[-2], qubits[-1])
            qc.cu3(np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.cu3(-np.pi/2, 0, 0, qubits[-2], qubits[-1])
            self._cnx(qc, *qubits[:-2], qubits[-1])
            qc.crz(-np.pi/2, qubits[-2], qubits[-1])
        elif len(qubits) == 3:
            qc.ccx(*qubits)
        elif len(qubits) == 2:
            qc.cx(*qubits)
        for i in qubits[self.subnodes:-1]:
            qc.x(i)

if __name__ == "__main__":
    test = QWonCircle(4, 4, 3)  # 2**(3), 
    count = test.QW()
    print(count)
