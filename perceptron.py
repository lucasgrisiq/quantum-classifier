import qiskit
from qiskit.algorithms.optimizers import SPSA
from noisyopt import minimizeSPSA
import numpy as np

from utils import train_df, test_df

_train_df = train_df()
_test_df = test_df()
train16_df = train_df(16)
test16_df = test_df(16)

class QPerceptron():
    
    def __init__(self, n):
        self.n = n
        self.qstates = qiskit.QuantumRegister(n, 'qr')
        self.input_states = _train_df.values if n == 10 else train16_df.values
        self.sim = qiskit.Aer.get_backend('qasm_simulator')

    def _nBin(self, x):
        return (''.join('0' for _ in range(self.n)) + bin(x)[2:])[-self.n:]

    def set_inputs(self, inputs):
        self.theta = inputs
    
    def set_weights(self, weights):
        self.phi = weights

    def _make_ui(self):
        q = qiskit.QuantumRegister(self.n, 'qr')
        ui = qiskit.QuantumCircuit(q)
        ui.h(q)

        prev_step = self._nBin(0)
        for s in range(1, np.power(2, self.n)):
            step = self._nBin(s)
            for i, state in enumerate(step):
                if state != prev_step[i]:
                    ui.x(q[i])
                
            th = self.theta[s] - self.theta[0]
            ui.mcrz(lam=th, q_controls=q[:-1], q_target=q[-1])

            prev_step = step

        return ui
    
    def _make_uw(self):
        q = qiskit.QuantumRegister(self.n, 'qr')
        uw = qiskit.QuantumCircuit(q)

        prev_step = self._nBin(0)
        for s in range(1, np.power(2, self.n)):
            step = self._nBin(s)
            for i, state in enumerate(step):
                if state != prev_step[i]:
                    uw.x(q[i])

            th = self.phi[s] - self.phi[0]
            uw.mcrz(lam=th, q_controls=q[:-1], q_target=q[-1])

            prev_step = step
            
        uw.h(q)
        uw.x(q)

        return uw

    def make_circuit(self):
        ansilla = qiskit.QuantumRegister(1, 'ansl')
        meas = qiskit.ClassicalRegister(1, 'cr0')
        circuit = qiskit.QuantumCircuit(self.qstates, ansilla, meas)

        ui = self._make_ui().to_gate(label='Ui')
        uw = self._make_uw().to_gate(label='Uw')

        circuit.append(ui, self.qstates)
        circuit.barrier()
        circuit.append(uw, self.qstates)
        circuit.barrier()

        circuit.mcx(self.qstates, ansilla)
        circuit.measure(ansilla, meas)

        return circuit

    def run(self, index: int, weights: list, threshold: float = 0.5):
        inputs = self.input_states[index][0]

        self.set_inputs(inputs)
        self.set_weights(weights)
        self.circuit = self.make_circuit()

        result = qiskit.execute(self.circuit, self.sim, shots=10).result().get_counts()
        resp = result['1'] / (result['1'] + result['0'])

        if resp > threshold:
            return 1
        else:
            return 0

    def cost_function(self, w: list):
        cost = 0
        for i in range(len(self.input_states)):
            y = self.run(i, w)
            cost += (self.input_states[i][1] - y) ** 2
        cost /= len(self.input_states)
        self.cost_array.append(cost)
        return cost

    def callback_spsa(self, xk):
        print('callback')
        print('Iter: {} | Custo: {}'.format(len(self.cost_array), self.cost_array[-1]))
        print(xk)

    def fit(self, maxiter: int = 300, learning_rate: float = 0.2):
        # spsa = SPSA(maxiter=maxiter)

        # result = spsa.optimize(
        #     num_vars = size,
        #     objective_function = self.cost_function,
        #     variable_bounds = [(0, np.pi/2)]*size,
        #     initial_point = [0]*size
        # )
        
        size = len(self.input_states[0][0])
        x0 = np.random.uniform(0, np.pi/2, size)
        self.cost_array = []

        res = minimizeSPSA(
            func=self.cost_function,
            x0=x0,
            bounds=[(0, np.pi/2)]*size,
            niter=maxiter,
            paired=False,   # sem seed
            c=0.1, # qtd de aleatoriedade  
            a=learning_rate,
            callback=self.callback_spsa,
            disp=True
        )

        print(res)
        return res