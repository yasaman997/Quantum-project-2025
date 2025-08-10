import pickle as pkl
import time
from pathlib import Path
from qiskit import qpy
import dataclasses
from qiskit_aer import AerSimulator
import docplex.mp.model_reader
from qiskit.compiler import transpile

ROOT = Path.cwd()
PROJECT_ROOT = ROOT / 'WISER_Optimization_VG'
print(f"Project Root identified as: {PROJECT_ROOT}")

import sys
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / '_experiments'))

from step1_26 import get_cplex_sol, problem_mapping
from experiment_py_new1 import Experiment
from step3_new5 import HardwareExecutor
from doe_localsearch_new2 import doe_localsearch as doe

def execute_multiple_runs(lp_file: str, experiment_id: str, num_exec: int, ansatz: str, ansatz_params: dict,
                          theta_initial: str, device: str, instance: str, optimizer: str, max_epoch: int, alpha: float, shots: int,
                          run_on_serverless: bool, theta_threshold: float):

    
    obj_fn, ansatz_, theta_initial_, backend, initial_layout = problem_mapping(
        lp_file, ansatz, ansatz_params, theta_initial, device, instance
    )

    refx, refval = get_cplex_sol(lp_file, obj_fn)
    (Path(lp_file).parent / experiment_id).mkdir(exist_ok=True)

    print("Preparing circuit for execution...")
    if device == 'AerSimulator':
        print("✅ Using AerSimulator. Translating ansatz to simulator's basis gates...")
        isa_ansatz = transpile(ansatz_, backend=backend)
    else:
        print(f"Targeting real hardware ({device}). Running preset pass manager...")
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        isa_ansatz = generate_preset_pass_manager(
            target=backend.target, optimization_level=3, initial_layout=initial_layout
        ).run(ansatz_)

    
    for exec_num in range(num_exec):
        experiment_id_with_exec = f'{experiment_id}/{exec_num}'
        out_file = Path(lp_file).parent / f'{experiment_id}/exp{exec_num}.pkl'

        if out_file.is_file():
            print(f'File {out_file} exists. Skipped.')
            continue

        print(f"Starting VQE run {exec_num+1}/{num_exec}...")
        t = time.time()
        he = HardwareExecutor(
            objective_fun=obj_fn,
            backend=backend,
            isa_ansatz=isa_ansatz,
            optimizer_theta0=theta_initial_,
            optimizer_method=optimizer,
            refvalue=refval,
            sampler_options={'default_shots':shots},
            use_session=False,
            iter_file_path_prefix=str(Path(lp_file).parent / experiment_id_with_exec),
            store_all_x=True,
            solver_options={"max_epoch": max_epoch, "alpha": alpha, 'theta_threshold': theta_threshold},
        )
        result = he.run()
        step3_time = time.time() - t

        out = Experiment.from_step3(
            experiment_id_with_exec,
            ansatz, ansatz_params, theta_initial, device, optimizer, alpha, theta_threshold, lp_file, shots, refx, refval,
            Experiment.get_current_classical_hw(), step3_time, he.job_ids,
            result, he.optimization_monitor
        )
        with open(out_file, 'bw') as f:
            pkl.dump(dataclasses.asdict(out), f)
        print(f"✅ VQE run {exec_num+1} complete. Results saved to {out_file}")



if __name__ == '__main__':
    qaoa_experiment_config = {
        'lp_file': str(PROJECT_ROOT / 'data/1/31bonds/docplex-bin-avgonly-nocplexvars.lp'),
        'experiment_id': 'QAOA1rep_piby3_AerSimulator_0.2_Final',
        'num_exec': 1,
        'ansatz': 'QAOA',
        'ansatz_params': {'reps': 1},
        'theta_initial': 'piby3',
        'device': 'AerSimulator',
        'optimizer': 'nft',
        'max_epoch': 100,
        'alpha': 0.2,
        'shots': 1000,
        'theta_threshold': 0.01
    }

    execute_multiple_runs(**qaoa_experiment_config, instance='', run_on_serverless=False)