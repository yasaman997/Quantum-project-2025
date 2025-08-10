# Quantum-project-2025

Project: Portfolio Optimization

Participant:
Name: Yasaman Yaghoobi

WISER Enrollment ID: gst-4XbircWh8ME6kEH

Project Presentation Deck:

https://github.com/yasaman997/Quantum-project-2025/blob/main/presentation-final.pdf


Project Summary 

At the heart of the financial industry lies a complex challenge: constructing the perfect portfolio. This is a delicate balancing act between maximizing returns, minimizing risk, and satisfying a vast web of constraints—a task that pushes even the most powerful classical computers to their limits. As portfolios swell to include thousands of assets, today's solvers can get bogged down, often finding a good solution but not necessarily the best one within the tight timeframes of a fast-paced market. This project asks: can we harness the power of quantum computing to find a better way?

My work began with the simplified model of portfolio construction challenge, focusing on a formulation with 
binary decision variables, linear constraints, and a quadratic objective function designed to minimize the tracking error against a target index . The initial goal was to establish a baseline by translating this constrained problem into a Quadratic Unconstrained Binary Optimization (QUBO) problem suitable for a quantum algorithm. I then implemented a series of targeted improvements to both the classical and quantum components of the workflow to enhance the quality of the final solution, focusing on the judging criteria of Optimality, Speed, and Scalability.

The first major improvement focused on the classical-quantum interface to improve Optimality. The standard method of converting constraints into penalties often uses a single, large penalty factor for all violations. This can create a difficult, "rugged" optimization landscape for the quantum algorithm to navigate. I replaced this with a more sophisticated per-constraint penalty mechanism. In this improved method, the penalty for violating each constraint is scaled by the L2-norm (or magnitude) of that constraint's own coefficients. This creates a more nuanced energy landscape, providing better guidance to the quantum optimizer and helping it find higher-quality solutions more efficiently.

The second key improvement targeted the quantum core of the algorithm to enhance both Optimality and Speed. Instead of using a generic, hardware-efficient ansatz, I implemented a problem-specific Quantum Approximate Optimization Algorithm (QAOA) ansatz. This circuit is constructed directly from the Hamiltonian of the optimization problem, meaning its structure inherently mirrors the interactions between the financial assets (bonds) in the portfolio. By using an ansatz that is tailored to the problem, the Variational Quantum Eigensolver (VQE) can often find better solutions with fewer parameters and in fewer iterations than a generic circuit would allow.

Finally, to further refine the Optimality of the solution, I enhanced the classical post-processing stage. The initial solution from the VQE is a strong candidate but is often a local minimum. To improve upon this, I implemented a powerful metaheuristic: a Variable Neighborhood Search (VNS). This algorithm systematically "shakes" a solution to escape local optima and then uses an intensive "best improvement" search to find the bottom of new, potentially deeper valleys in the solution space.

The resulting end-to-end pipeline—from the improved problem formulation and custom QAOA ansatz to the advanced classical polishing—represents a robust and sophisticated framework for benchmarking hybrid quantum solutions. The final quantum solution is validated by ensuring its bitstring satisfies the original problem constraints, and its performance is compared directly against the objective value obtained from a time-limited run of a classical CPLEX solver, providing a clear measure of success.

Challenge Deliverables
Here is a breakdown of how each of the challenge deliverables was addressed.

1. Review of the Mathematical Formulation
The project focuses on the simplified portfolio creation model of an index-tracking bond fund. As specified in the project documentation, the mathematical formulation consists of:

Binary Decision Variables: A set of binary variables y_c where y_c = 1 if bond c is included in the portfolio, and 0 otherwise.

Quadratic Objective Function: The goal is to minimize a quadratic objective function that measures the squared distance between the characteristics of the selected portfolio and a predefined target, summed over all risk groups and characteristics.

Linear Constraints: The optimization is subject to several linear constraints, including a limit on the total number of bonds and risk factor guardrails .

2. Conversion to a Quantum-Compatible Formulation
To make the problem compatible with a quantum algorithm, the constrained optimization problem was converted into an unconstrained format (a QUBO). This was achieved by incorporating the linear constraints as penalty terms in the objective function. A novel per-constraint penalty mechanism was implemented, where the penalty for violating each constraint is scaled by that constraint's magnitude, creating a more effective optimization landscape.

The following code snippet demonstrates this conversion:

Code Logic:

```python
# Calculate a base penalty factor based on the objective's range
max_obj = np.sum(Q, where=Q>0)
min_obj = np.sum(Q, where=Q<0)
base_penalty = (max_obj - min_obj) * 1.1

# Create an array of penalties, one for each constraint
penalties = np.zeros(num_ctr)
for i in range(num_ctr):
    # Scale each penalty by the magnitude (L2-norm) of its constraint's coefficients
    constraint_magnitude = np.linalg.norm(A[i, :])
    penalties[i] = base_penalty * constraint_magnitude if constraint_magnitude > 0 else base_penalty

# The final objective function combines the original goal with the intelligent penalties
def obj_fn_embedding_constraints(x):
    obj_val = x @ (sparseQ @ x)
    violations = np.maximum(b - (sparseA @ x), 0)**2
    penalty_term = np.sum(penalties * violations)
    return obj_val + c + penalty_term
```

3. Quantum Optimization Program (VQE with QAOA Ansatz)
The quantum program was written using the Variational Quantum Eigensolver (VQE). For the quantum circuit, a problem-specific QAOA (Quantum Approximate Optimization Algorithm) ansatz was implemented. This circuit is constructed directly from the problem's Hamiltonian, making it more effective than a generic ansatz.

The build_qaoa_ansatz function builds this custom circuit:


```
def build_qaoa_ansatz(hamiltonian_terms, reps=1):

    h, J = hamiltonian_terms
    num_qubits = len(h)
    
    # Create the cost operator (problem Hamiltonian) from the problem's h and J terms
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

    # Build and decompose the QAOA ansatz circuit
    qaoa_ansatz = QAOAAnsatz(cost_operator, reps=reps, mixer_operator=mixer_operator)
    return qaoa_ansatz.decompose()
```

4. Solving the Optimization Problem
The VQE algorithm was implemented using Qiskit's standard library. The VQE class combines the QAOA ansatz, a COBYLA optimizer, and the AerSimulator to iteratively find the lowest energy eigenvalue of the problem's Hamiltonian, which corresponds to the optimal solution of the portfolio problem.

5. Validation with a Classical Routine
The final, polished solution from the Variable Neighborhood Search was validated using a classical routine to ensure its quality and correctness. The validation process involved two key checks:

Feasibility: The final bitstring representing the selected portfolio was checked against the original set of linear constraints to confirm that all business and risk rules were satisfied.

Quality Benchmark: The final objective function value of the polished solution was compared against the benchmark value obtained from a time-limited (60-second) run of a professional classical solver (CPLEX) on the same problem.

6. Comparison of Solution Quality and Performance
The performance of the workflow was evaluated by comparing the solution quality before and after the classical polishing step. The results demonstrated a clear and significant improvement achieved by the Variable Neighborhood Search (VNS).

Initial Solution: The initial candidate solution had an objective value of 40.3934. This is a strong result, but it represents a local minimum in the optimization landscape.

Final Polished Solution (after VNS): The Variable Neighborhood Search took this initial result and successfully found a better solution, reducing the objective value to 40.3047.

This demonstrates the power of using advanced classical heuristics to refine an initial solution. The VNS was able to navigate the solution space more effectively, escaping the initial local optimum to find a superior portfolio configuration. When compared to the time-limited classical benchmark of 40.2939, our final polished result is highly competitive, showcasing the effectiveness of this classical refinement strategy. Performance metrics such as the runtime for the fast VNS polishing step are tracked to evaluate the overall efficiency of the solution.
