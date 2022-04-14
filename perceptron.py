import qiskit
import numpy as np

class QPerceptron():
    
    def __init__(self, n, theta, phi):
        self.n = n
        self.theta = theta
        self.phi = phi
        self.qstates = qiskit.QuantumRegister(n, 'qr')
        self.circuit = self.make_circuit()
    
    def _nBin(self, x):
        return (''.join('0' for _ in range(self.n)) + bin(x)[2:])[-self.n:]

    def set_theta(self, theta):
        self.theta = theta
    
    def set_phi(self, phi):
        self.phi = phi

    def make_ui(self):
        q = qiskit.QuantumRegister(self.n, 'qr')
        ui = qiskit.QuantumCircuit(q)
        ui.h(q)

        for s in range(1, np.power(2, self.n)):
            step = self._nBin(s)
            apply_x = np.zeros(self.n)
            for i, pos in enumerate(step):
                if pos == '0':
                    ui.x(q[i])
                    apply_x[i] = 1

            th = self.theta[s] - self.theta[0]
            ui.mcrz(lam=th, q_controls=q[:-1], q_target=q[-1])

            for p, x in enumerate(apply_x):
                if x == 1:
                    ui.x(q[p])
        return ui
    
    def make_uw(self):
        q = qiskit.QuantumRegister(self.n, 'qr')
        uw = qiskit.QuantumCircuit(q)
        
        for s in range(1, np.power(2, self.n)):
            step = self._nBin(s)
            apply_x = np.zeros(self.n)
            for i, pos in enumerate(step):
                if pos == '0':
                    uw.x(q[i])
                    apply_x[i] = 1

            th = self.phi[s] - self.phi[0]
            th = self.theta[s] - self.theta[0]

            controls = [q[i] for i in range(self.n - 1)]
            uw.mcrz(lam=th, q_controls=controls, q_target=q[-1])

            for p, x in enumerate(apply_x):
                if x == 1:
                    uw.x(q[p])
            
        uw.h(q)
        uw.x(q)

        return uw

    def make_circuit(self):
        ansilla = qiskit.QuantumRegister(1, 'ansl')
        meas = qiskit.ClassicalRegister(1, 'cr0')
        circuit = qiskit.QuantumCircuit(self.qstates, ansilla, meas)

        ui = self.make_ui().to_gate(label='Ui')
        uw = self.make_uw().to_gate(label='Uw')

        circuit.append(ui, self.qstates)
        circuit.barrier()
        circuit.append(uw, self.qstates)
        circuit.barrier()

        circuit.mcx(self.qstates, ansilla)
        circuit.measure(ansilla, meas)

        return circuit

    def draw_ui(self):
        print(self.make_ui().draw())

    def draw_uw(self):
        print(self.make_uw().draw())

    def draw_circuit(self):
        print(self.circuit.draw())


if __name__ == "__main__":
    n = 3
    theta = np.random.uniform(0, 2*np.pi, size=8)
    phi = np.random.uniform(0, 2*np.pi, size=8)

    p = QPerceptron(n, theta, phi)
    p.draw_circuit()