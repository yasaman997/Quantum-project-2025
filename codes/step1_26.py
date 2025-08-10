import numpy as np
import docplex.mp
import docplex.mp.model
import docplex.mp.constants
from qiskit.circuit.library import TwoLocal, NLocal, RYGate, QAOAAnsatz
from qiskit.circuit import QuantumCircuit, ParameterVector, Measure, Parameter
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import CouplingMap, Target
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import CXGate, RZGate, RXGate, RYGate, HGate, IGate
import numba
from numba import jit

from scipy.sparse import csr_array

def model_to_obj_sparse_numba(model: docplex.mp.model.Model):
    num_vars = model.number_of_binary_variables
    num_ctr = model.number_of_constraints
    print(num_vars, num_ctr)
    Q = np.zeros((num_vars, num_vars))
    c = model.objective_expr.get_constant()
    for i, dvari in enumerate(model.iter_variables()):
        for j, dvarj in enumerate(model.iter_variables()):
            if i > j: continue
            Q[i,j] = model.objective_expr.get_quadratic_coefficient(dvari, dvarj)
            if i == j:
                Q[i,i] += model.objective_expr.linear_part.get_coef(dvari)
    A = np.zeros((num_ctr, num_vars))
    b = np.zeros(num_ctr)
    for i, ctr in enumerate(model.iter_constraints()):
        sense = 1 if ctr.sense == docplex.mp.constants.ComparisonType.GE else -1
        for j, dvarj in enumerate(model.iter_variables()):
            A[i,j] = sense * ctr.lhs.get_coef(dvarj)
        b[i] = sense * ctr.rhs.get_constant()
    max_obj = np.sum(Q, where=Q>0)
    min_obj = np.sum(Q, where=Q<0)
    base_penalty = (max_obj - min_obj) * 1.1
    penalties = np.zeros(num_ctr)
    for i in range(num_ctr):
        constraint_magnitude = np.linalg.norm(A[i, :])
        if constraint_magnitude > 0:
            penalties[i] = base_penalty * constraint_magnitude
        else:
            penalties[i] = base_penalty
    sparseQ = csr_array(Q)
    sparseA = csr_array(A)
    h = np.diag(Q).copy()
    J = {}
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if not np.isclose(Q[i, j], 0):
                J[(i, j)] = Q[i, j]
    def obj_fn_embedding_constraints(x):
        obj_val = x @ (sparseQ @ x)
        violations = np.maximum(b - (sparseA @ x), 0)**2
        penalty_term = np.sum(penalties * violations)
        return obj_val + c + penalty_term
    return obj_fn_embedding_constraints, (h, J)

model_to_obj = model_to_obj_sparse_numba

def build_qaoa_ansatz(hamiltonian_terms, reps=1):
    h, J = hamiltonian_terms
    num_qubits = len(h)
    pauli_list = []
    for i in range(num_qubits):
        if not np.isclose(h[i], 0):
            pauli_string = ['I'] * num_qubits
            pauli_string[i] = 'Z'
            pauli_list.append(("".join(pauli_string), h[i]))
    for (i, j), coeff in J.items():
        if not np.isclose(coeff, 0):
            pauli_string = ['I'] * num_qubits
            pauli_string[i] = 'Z'
            pauli_string[j] = 'Z'
            pauli_list.append(("".join(pauli_string), coeff))
    cost_operator = SparsePauliOp.from_list(pauli_list)
    mixer_operator = SparsePauliOp.from_list([("X" * num_qubits, 1)])
    qaoa_ansatz = QAOAAnsatz(cost_operator, reps=reps, mixer_operator=mixer_operator)
    return qaoa_ansatz.decompose()

def get_cplex_sol(lp_file: str, obj_fn):
    model = docplex.mp.model_reader.ModelReader.read(lp_file)
    sol = model.solve()
    x_cplex = [v.solution_value for v in model.iter_binary_vars()]
    return x_cplex, sol.objective_value

def build_ansatz(ansatz: str, ansatz_params: dict, num_qubits: int, backend: BackendV2) -> tuple[QuantumCircuit, dict | None]:
    pass

def get_backend(device: str, instance: str, num_vars: int) -> BackendV2:
    print(f"--- [DEBUG] get_backend called for device: {device} with num_vars: {num_vars} ---")
    if device == 'AerSimulator':

        return AerSimulator(method='matrix_product_state')
        # --- END FIX ---
    elif device.startswith('ibm_'):
        service = QiskitRuntimeService()
        return service.backend(device, instance)
    else:
        raise ValueError('unknown device')

def problem_mapping(lp_file: str, ansatz: str, ansatz_params: dict, theta_initial: str, device: str, instance: str):
    model = docplex.mp.model_reader.ModelReader.read(lp_file)
    obj_fn, hamiltonian_terms = model_to_obj(model)
    num_vars = model.number_of_binary_variables
    backend = get_backend(device, instance, num_vars)
    if ansatz == 'QAOA':
        ansatz_ = build_qaoa_ansatz(hamiltonian_terms, reps=ansatz_params.get('reps', 1))
        initial_layout = None
    else:
        ansatz_, initial_layout = build_ansatz(ansatz, ansatz_params, num_vars, backend)
    
    ansatz_.measure_all(inplace=True)
    
    if theta_initial == 'piby3':
        theta_initial_ = np.pi/3 * np.ones(ansatz_.num_parameters)
    else:
        raise ValueError('unknown theta_initial')
    return obj_fn, ansatz_, theta_initial_, backend, initial_layout