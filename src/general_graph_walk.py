from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import numpy as np

q = QuantumRegister(3)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)

for step in range(1):
    qc.h(q[2])
    # qc.ccx(q[2], q[1], q[0])
    qc.cx(q[2], q[0])
    qc.cx(q[2], q[1])
    qc.x(q[1])

    # qc.x(q[2])
    # qc.cx(q[2], q[0])
    # qc.cx(q[2], q[1])
    # qc.x(q[2])
    # qc.x(q[1])

qc.measure(q[0], c[0])
qc.measure(q[1], c[1])

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend=backend, shots=1024)
count = job.result().get_counts(qc)
print(qc)
print(count)