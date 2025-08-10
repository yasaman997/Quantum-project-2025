
from __future__ import annotations

import logging
from typing import Any, Callable, Tuple, Optional, Union, Dict

import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.result import Result
from qiskit_ibm_runtime import SamplerV2, Session, SamplerOptions
from qiskit.providers.exceptions import QiskitError
from scipy.optimize import OptimizeResult
from numba import jit


from optimization9 import BestValueMonitor, OptimizationMonitor, uncompress_x 
from sbo.src.optimizer.optimization_wrapper import run as run_wrapper

LOGGER = logging.getLogger(__name__)


class HardwareExecutor:

    def __init__(
        self,
        objective_fun: Callable[..., float],
        backend: BackendV2,
        isa_ansatz: QuantumCircuit,
        optimizer_theta0: np.ndarray[Any, Any] | None = None,
        optimizer_method: str | Callable = "nft",
        sampler_result_to_aggregate_objective: Callable[[Result], float] | None = None,
        refvalue: float | None = None,
        sampler_options: Optional[Union[Dict, SamplerOptions]] = {},
        use_session: bool = True,
        verbose: str | None = None,
        file_name: str | None = None,
        store_all_x: bool = False,
        iter_file_path_prefix: str | None = None,
        solver_options: dict | None = None,
        max_retries: int = 10,
        run_id: str | None = None,
    ):
        self.backend = backend
        self.isa_ansatz = isa_ansatz
        self.optimizer_theta0 = optimizer_theta0
        self.optimizer_method = optimizer_method
        self.refvalue = refvalue
        self.sampler_options = sampler_options
        self.verbose = verbose
        self.file_name = file_name
        self.solver_options = solver_options
        self.max_retries = max_retries
        self.run_id = str(int(time.time())) if run_id is None else run_id
        self.objective_fun = objective_fun

        self.objective_monitor = BestValueMonitor(objective_fun, store_all_x)

        self.optimization_monitor = OptimizationMonitor(
            self.calc_aggregate_objective,
            self.objective_monitor,
            verbose=verbose,
            refvalue=refvalue,
            file_name=file_name,
            iter_file_path_prefix=iter_file_path_prefix,
        )

        self.job_ids: list[str] = []

        if sampler_result_to_aggregate_objective is None:
            self._sampler_result_to_aggregate_objective = self._sampler_result_to_cvar
        else:
            self._sampler_result_to_aggregate_objective = sampler_result_to_aggregate_objective


        if use_session:
            self._run_method = self._run_with_session
        else:
            self._run_method = self._run_with_jobs

    def _run_with_sampler(self) -> OptimizeResult:
        self.sampler.options.environment.job_tags = [self.run_id, f'{self.run_id}_{self.optimization_monitor.number_of_iterations()}']
        result = run_wrapper(
            isa_ansatz=self.isa_ansatz,
            optimizer_x0=self.optimizer_theta0,
            optimization_fun=self.optimization_monitor.cost,
            optimizer_args=(),
            optimizer_method=self.optimizer_method,
            optimizer_callback=self.optimization_monitor.callback,
            solver_options=self.solver_options,
        )
        return result

    def _run_with_session(self) -> OptimizeResult:
        with Session(backend=self.backend) as session:
            self.sampler = SamplerV2(mode=session, options=self.sampler_options)
            return self._run_with_sampler()

    def _run_with_jobs(self) -> OptimizeResult:
        self.sampler = SamplerV2(mode=self.backend, options=self.sampler_options)
        return self._run_with_sampler()

    def run(self) -> OptimizeResult:
        try:
            result = self._run_method()
        except QiskitError as error:
            result = OptimizeResult()
            if self.optimization_monitor.callback_count > 0:
                gtheta, theta = self.optimization_monitor.best_seen_result()
                result.x = theta
                result.fun = gtheta
            result.nit = self.optimization_monitor.number_of_iterations()
            result.fevals = self.optimization_monitor.number_of_function_evaluations()
            result.success = False
            result.message = str(error)
            return result
        return result
    
    def submit_job(self, theta, remaining_attempts: int):
        job = self.sampler.run([(self.isa_ansatz, theta)])
        job_id = job.job_id() if callable(job.job_id) else job.job_id
        self.job_ids.append(job_id)
        try:
            return job.result()
        except QiskitError as error:
            if remaining_attempts > 0:
                LOGGER.warning('job %s failed, retrying (%d remaining attempts)', job_id, remaining_attempts)
                return self.submit_job(theta, remaining_attempts=remaining_attempts-1)
            else:
                raise error

    def calc_aggregate_objective(self, theta: np.ndarray) -> float:
        self.optimization_monitor.objective_monitor.list_x.append(self.optimization_monitor.objective_monitor.list_job_x)
        self.optimization_monitor.objective_monitor.list_cnt.append(self.optimization_monitor.objective_monitor.list_job_cnt)
        self.optimization_monitor.objective_monitor.list_fx.append(self.optimization_monitor.objective_monitor.list_job_fx)
        self.optimization_monitor.objective_monitor.list_job_x = []
        self.optimization_monitor.objective_monitor.list_job_cnt = []
        self.optimization_monitor.objective_monitor.list_job_fx = []
        result = self.submit_job(theta, self.max_retries)

        aggr_obj = self._sampler_result_to_aggregate_objective(result)
        return aggr_obj
    
    @staticmethod
    @jit
    def calc_cvar(sorted_count: np.ndarray, sorted_fx: np.ndarray, ak: int) -> float:
        count_cumsum = 0
        cvar = 0
        for idx, cnt in np.ndenumerate(sorted_count):
            count_cumsum += cnt
            cvar += cnt * sorted_fx[idx]
            if count_cumsum >= ak:
                break
        cvar -= sorted_fx[idx]*(count_cumsum-ak)
        return cvar / ak

    def _sampler_result_to_cvar(self, result: Result) -> float:
     
        alpha = self.solver_options.get("alpha", 1.0)

        counts = result[0].data['counts']
        total_shots = sum(counts.values())
     

        dtype = [('fx', float), ('cnt', int)]
        vals = np.array([
            (
                self.objective_fun(np.array(list(bins[::-1]), dtype=float)), 
                cnt
            )
            for bins, cnt in counts.items()
            ], dtype=dtype)
        vals.sort(kind='heapsort', order='fx')
        
        ak = int(np.ceil(total_shots * alpha))

        cvar = HardwareExecutor.calc_cvar(vals['cnt'], vals['fx'], ak)
        return cvar