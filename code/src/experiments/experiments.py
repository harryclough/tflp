from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from multiprocessing import Value
import os
import json
from typing import Optional, Any, Generator

from src.colgensolver import ColgenSolver
from src.experiments.result import ExperimentResult, save_metadata
from src.graph import *
from src.instance import *
from src.pathinstance import PathInstanceParams, new_path_instance
from src.scipysolver import ScipySolver
from src.experiments.objectives import *


class Verbosity:
    NONE = 0
    LOW  = 1
    HIGH = 2


class Experiment:
    def __init__(self, params: InstanceParams, directed: bool, name: str,
                 var_names: List[str], meta_vars: dict, verbosity: int, save: bool, save_path: str,
                 new_solver=ScipySolver, new_inst=new_instance,
    ):
        self.params = params
        self.directed = directed
        self.name = name
        self.var_names = var_names
        self.meta_vars = meta_vars
        self.verbosity = verbosity
        
        self.solver_verbose = (self.verbosity >= Verbosity.HIGH)
        
        self.new_inst = new_inst
        self.new_solver = new_solver
        
        self.save = save
        self.save_path = save_path
        if save:
            if not save_path:
                raise ValueError("Save is enabled but no save path is provided.")
            elif not os.path.exists(save_path):
                raise FileNotFoundError(f"Invalid path: {save_path}")
            else:
                i = 0
                path = ""
                while i == 0 or os.path.exists(path):
                    path = os.path.join(save_path, f"{self.name}_{i}")
                    i += 1
                self.save_path = path
                os.mkdir(self.save_path)
    
    def get_save_name(self, edge: Optional[Edge], iter: int) -> str:
        if edge:
            return f"{self.name}_({edge[0]}-{edge[1]})_i-{iter}"
        else:
            return f"{self.name}_(Control)_i-{iter}"
    
    def run(self, iters = 1) -> List[ExperimentResult]:
        raise NotImplementedError("Abstract method called.")

    def run_control(self, experiment_vars: list, iter: int) -> ExperimentResult:
        if self.verbosity >= Verbosity.LOW:
            print(f"\nREMOVING: None (Control)")
        inst = self.new_inst(self.params)
        solver = self.new_solver(inst)
        solver.solve(verbose=self.solver_verbose)
        
        return self.create_result(solver, experiment_vars, iter)

    def create_result(self, solver, experiment_vars: list, iteration: int) -> ExperimentResult:
        return ExperimentResult(solver, self.name, experiment_vars.copy(), iteration)


class BruteForceEdges(Experiment):
    def __init__(self, params: InstanceParams, name: str,
                 verbosity=Verbosity.NONE, directed=True, save=False, save_path="",
                 meta_vars:dict={}, new_inst=new_instance, new_solver:Any=ScipySolver,
    ):
        super().__init__(
            params, directed, name, ["EDGE_REMOVED"], meta_vars, verbosity, save, save_path,
            new_inst=new_inst, new_solver=new_solver,
        )
    
    def run(self, iters = 1, edges_to_remove:Union[List[Edge], None]=None) -> List[ExperimentResult]:
        if not edges_to_remove:
            edges_to_remove = list(self.params.car_graph.edges())
            
        results = []
        for iter in range(iters):
            if self.verbosity >= Verbosity.LOW:
                print(f"\nRunning '{self.name}' ({iter+1} / {iters})...")
            result = self.run_control(["None"], iter)
            # TODO: Clean this up (maybe move into run_control?)
            if self.save:
                result.save(self.save_path, self.get_save_name(None, iter))
                    
            results.append(result)
            removed_edges = set()
            for edge in edges_to_remove:
                if apply_directed(edge, self.directed) in removed_edges:
                    continue
                try:
                    result = self.eval_without_edge(iter, edge)
                except GraphError as e:
                    print("WARN: GraphError while removing edge:", e)
                    print("Skipping edge.")
                    continue
                results.append(result)
                removed_edges.add(apply_directed(edge, self.directed))
                if self.save:
                    result.save(self.save_path, self.get_save_name(edge, iter))
                if self.verbosity >= Verbosity.HIGH:
                    print(result)
        if self.verbosity >= Verbosity.LOW:
            print(f"\nExperiment '{self.name}' complete.")
        if self.save:
            p = os.path.join(self.save_path, f"{self.name}_metadata.json")
            save_metadata(p, self.name, self.var_names, len(results), iters, self.meta_vars)
        return results
    
    def eval_without_edge(self, iter: int, edge: Edge) -> ExperimentResult:
        old_graph = self.params.car_graph
        self.params.car_graph = old_graph.copy()
        self.params.car_graph.remove_edge(edge)
        if not self.directed:
            self.params.car_graph.remove_edge(flip_edge(edge))
        if self.verbosity >= Verbosity.LOW:
            print(f"\nREMOVING: {edge}")
        
        inst = self.new_inst(self.params)
        solver = self.new_solver(inst)
        try:
            solver.solve(verbose=self.solver_verbose)
        finally:
            self.params.car_graph = old_graph
        
        return self.create_result(solver, [edge], iter)


def generate_var_combinations(default_vals: Dict, overrides: List[Dict]) -> List[Dict]:
    if not overrides:
        return [default_vals]
    return [{**default_vals, **override} for override in overrides]


def get_varied_params(
    adjs: List[AdjacencyGraph],
    demand_fs: List[Callable[[], float]],
    variables: List[Dict],
    n_paths: int,
) -> List[InstanceParams]:
    ls = []
    for adj, d_f, vs in product(adjs, demand_fs, variables):
        ls.append(InstanceParams(adj.copy(), adj.copy(), vs, n_paths, d_f, None))
    return ls

def get_varied_path_params(
    adjs: List[AdjacencyGraph],
    demand_fs: Union[List[Callable[[], float]], None],
    variables: List[Dict],
    max_car_paths: int,
    n_bus_paths: int,
    demands: Union[Demands, None],
) -> List[InstanceParams]:
    ls = []
    if demand_fs:
        for adj, d_f, vs in product(adjs, demand_fs, variables):
            ls.append(PathInstanceParams(adj.copy(), adj.copy(), vs, None, d_f, None, max_car_paths))
    else:
        for adj, vs in product(adjs, variables):
            ls.append(PathInstanceParams(adj.copy(), adj.copy(), vs, None, None, demands, max_car_paths))
    return ls


def print_cols(n_spaces, *args):
    print("".join([str(arg).ljust(n_spaces) for arg in args]))


def pprint_solution(inst: Instance, vars):
    COL_W = 12
    vars[vars <= 0.0] = 0.0  # Remove "-0.0"
        
    print("\nFraction of people on each path:")
    for (s, d) in inst.source_dest_pairs:
        paths = inst.car_routes[(s, d)]
        total = sum(vars[path.id] for path in paths)
        print(f"{s} -> {d}")
        for path in paths:
            print_cols(COL_W, f"{vars[path.id]:.2f}", path.node_list)
        print_cols(COL_W, f"= {total:.2f}", "Total")
        print()

def main():
    demand_fn = lambda: 4.0
    adj = get_graph(QUAD_GRID)
    
    variables = {
        "alpha_car": 1.0,
        "beta_car": 1.0,
        "zeta_car": (1.0, 1.0),
        "alpha_bus": 1.0,
        "beta_bus": 1.0,
        "zeta_bus": (1.0, 1.0),
        "zeta_none": (-1000.0, 1.0),
    }
    
    N_PATHS = 5
    
    # params = InstanceParams(
    #     adj.copy(), adj.copy(), demand_fn, variables, N_PATHS,
    # )
    # experiment = EdgeRemovalBF(
    #     params, "EdgeRemovalExp", verbosity=Verbosity.HIGH, directed=True, save=True, save_path="results"
    # )
    params = PathInstanceParams(
        adj.copy(), adj.copy(), variables, N_PATHS, demand_fn, None, 100,
    )
    experiment = BruteForceEdges(
        params, "EdgeRemovalExp", verbosity=Verbosity.HIGH, directed=True, save=True, save_path="results/experiments-main",
        new_inst=new_path_instance, new_solver=ColgenSolver,
    )
    
    results = experiment.run(1)

if __name__ == "__main__":
    main()
