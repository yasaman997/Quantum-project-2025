from __future__ import annotations


"""Class to monitor the execution of optimization."""
import logging
from typing import Any, Callable

import numpy as np
import pickle
import asyncio

LOGGER = logging.getLogger(__name__)

def compress_x(bitstring: np.ndarray) -> int:
    return int(''.join([str(int(z)) for z in bitstring]), 2)

def uncompress_x(compressed: int, num_qubits: int) -> np.ndarray:
   
    binary_string = format(compressed, 'b').zfill(num_qubits)
    return np.fromiter(map(int, binary_string), dtype='float')
    
class RefValueReached(Exception):
    """Custom exception for stopping the algorithm when refvalue was found."""

    def __init__(self, theta: np.ndarray, result: float, nit: int, nfev: int):
        self.theta = theta
        self.result = result
        self.nit = nit
        self.nfev = nfev
    
    pass


class BestValueMonitor():
    """
    Keep best objective value result and parameters.
    """

    def __init__(self, f: Callable, store_all_x: bool = False):
        """
        Constructor for BestValueMonitor.
        Args:
            f: function to be wrapped.
        """
        self._f: Callable = f
        self._store_all_x: bool = store_all_x

        self.best_x: np.ndarray | None = None
        self.best_fx: float | None = None
        self.list_best_iter: list[int] = [] 
        self.list_best_x: list[np.ndarray] = []
        self.list_best_fx: list[float] = [] 
        self.list_job_x: list[np.ndarray] = []
        self.list_job_cnt: list[int] = []
        self.list_job_fx: list[float] = []
        self.list_x: list[list[np.ndarray]] = []
        self.list_cnt: list[list[int]] = []
        self.list_fx: list[list[float]] = []
        self.iter_best_x: np.array = None 
        self.iter_best_fx: float = None 
        self.iter_fx_evals: int = 0 


    def cost(self, x: np.ndarray[Any, Any], *args: dict, iter=-1, cnt=-1):
        """Executes the actual simulation and returns the result, while
        keeping track of best value for the result
        Args:
            x (np.array): parameters for the function
            *args: additional arguments
        Returns:
            result of the function call
        """
       
        result = self._f(x, *args)
        self.iter_fx_evals += 1

        if self._store_all_x:
            self.list_job_x.append(x)
            self.list_job_cnt.append(cnt)
            self.list_job_fx.append(result)

        if self.iter_best_fx is None or result < self.iter_best_fx:
            self.iter_best_fx = result
            self.iter_best_x = x
        
        if self.best_fx is None or result < self.best_fx:
            self.best_fx = result
            self.best_x = x
            self.list_best_iter.append(iter)
            self.list_best_fx.append(result)
            self.list_best_x.append(x)

        return result

    def best_fx_x(self):
        """Returns best seen result in calls

        Returns:
            tuple: best result and best parameters
        """
        if self.best_fx is None:
            raise RuntimeError("function is not yet optimized")

        return self.best_fx, self.best_x


class OptimizationMonitor:
    """Avoid repeated calls for callback
    Stores history
    """

    def __init__(  
        self,
        function: Callable,
        objective_monitor: BestValueMonitor,
        verbose: str | None = None,
        refvalue: float | None = None,
        file_name: str | None = None,
        iter_file_path_prefix: str | None = None
    ):
        self._f = function
        if verbose is None:
            self._verbose = ""
        else:
            self._verbose = verbose
        self.refvalue = refvalue
        self.calls_count = 0
        self.callback_count = 0
        self.objective_monitor = objective_monitor
        self.list_calls_inp: list[np.ndarray] = []
        self.list_calls_res: list[float] = []
        self.list_callback_inp: list[np.ndarray] = []
        self.list_callback_res: list[float] = []
        self.list_callback_monitor_best: list[float] = []
        self.iter_best_x: list[np.ndarray] = []
        self.iter_best_fx: list[float] = []
        self.iter_fx_evals: list[int] = []
        self.logger = None
        if file_name is not None:
            self.logger = logging.getLogger("OptimizationMonitor")
            self.logger.setLevel(logging.INFO)

            handler = logging.FileHandler(file_name)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        
        self._iter_file_path_prefix = iter_file_path_prefix


    def cost(self, theta: np.ndarray, *args: dict):
        if self._verbose == "cost":
            print(f"cost: {self.calls_count:10d}", theta)
        result = self._f(theta, *args)
        self.list_calls_inp.append(theta)
        self.list_calls_res.append(result)
        self.calls_count += 1
        return result

    @staticmethod
    async def write_iter(file_name, data):
        data['list_x'] = [[compress_x(y) for y in x] for x in data['list_x']]
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def callback(self, theta: np.ndarray, *args, **kwargs: dict):
        LOGGER.info("optimizer internal status: %s %s", args, kwargs)
        if self._iter_file_path_prefix is not None:
            file_name = f'{self._iter_file_path_prefix}_{self.number_of_iterations()}.pkl'
            data = {
                'optimizer_internal_status': args,
                'optimizer_internal_status_dict': kwargs,
                'list_x': self.objective_monitor.list_x,
                'list_cnt': self.objective_monitor.list_cnt,
                'list_fx': self.objective_monitor.list_fx,
                }
            asyncio.run(OptimizationMonitor.write_iter(file_name, data))

        self.objective_monitor.list_x = []
        self.objective_monitor.list_fx = []
        self.objective_monitor.list_cnt = []
        self.iter_best_x.append(self.objective_monitor.iter_best_x)
        self.iter_best_fx.append(self.objective_monitor.iter_best_fx)
        self.iter_fx_evals.append(self.objective_monitor.iter_fx_evals)
        self.objective_monitor.iter_best_x = None
        self.objective_monitor.iter_best_fx = None
        self.objective_monitor.iter_fx_evals = 0

        if self._verbose == "callback":
            print(f"cabk: {self.callback_count:10d}", theta)
        
        theta = np.atleast_1d(theta)
        found = False
        i = 0
        for i, theta0 in reversed(list(enumerate(self.list_calls_inp))):
            theta0 = np.atleast_1d(theta0)
            if np.allclose(theta0, theta):
                found = True
                break
        
        if found:
            self.list_callback_inp.append(theta)
            self.list_callback_res.append(self.list_calls_res[i])
            self._add_best_value(theta, i)
        else:
            LOGGER.info("Unexpected behavior: Result of optimizer cost function call not found.")
            self.list_callback_inp.append(self.list_calls_inp[-1])
            self.list_callback_res.append(self.list_calls_res[-1])
            self._add_best_value(theta, i)

        self.callback_count += 1

    def _add_best_value(self, theta: np.ndarray, i: int):
        if self.objective_monitor is not None:
            best_fx = self.objective_monitor.best_fx_x()[0]
            self.list_callback_monitor_best.append(best_fx)
            if self.refvalue is not None:
                if np.isclose(best_fx, np.float64(self.refvalue)) or best_fx < self.refvalue:
                    raise RefValueReached(
                        theta,
                        self.list_calls_res[i],
                        self.number_of_iterations(),
                        self.number_of_function_evaluations(),
                    )

    def best_seen_result(self):
        if self.callback_count == 0:
            raise RuntimeError("function is not yet optimized")
        min_index = self.list_callback_res.index(min(self.list_callback_res))
        min_value = self.list_callback_res[min_index]
        corresponding_parameters = self.list_callback_inp[min_index]
        return min_value, corresponding_parameters

    def number_of_function_evaluations(self):
        return self.calls_count

    def number_of_iterations(self):
        return self.callback_count