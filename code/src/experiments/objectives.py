import numpy as np
from typing import List, Sequence
from pandas import DataFrame

from src.experiments.result import ExperimentResult
from src.scipysolver import ScipySolver


class Objective:
    def __init__(self, name: str):
        self.name = name
    
    def collect(self, solver: ScipySolver) -> float:
        raise NotImplementedError("Abstract method called.")


def collect_to_df(var_names: dict, results: List[ExperimentResult], metrics: Sequence[Objective]) -> DataFrame:
    header = (["iter"] +
        [v for v in var_names] +
        [m.name for m in metrics])
    data = []
    for result in results:
        row = [result.iteration]
        row += [v for v in result.experiment_vars]
        row += [m.collect(result.solver) for m in metrics]
        data.append(row)
    return DataFrame(data, columns=header)


class MeanTravelTime(Objective):
    NAME = "Congestion"
    
    def __init__(self):
        super().__init__(self.NAME)
    
    def collect(self, solver: ScipySolver) -> float:
        if solver.last_xs is None:
            raise ValueError("Solver has no vars.")
        inst = solver.inst
        
        congestion = 0.0
        base_latency = 0.0
        
        flows = inst.get_edge_flows(solver.last_xs)

        for (s, d) in inst.source_dest_pairs:
            car_lat = min(inst.get_car_latencies(s, d, flows))
            bus_lat = min(inst.get_bus_latencies(s, d, flows))
            base_car_lat = min(p.length for p in inst.car_routes[(s, d)])
            base_bus_lat = min(p.length for p in inst.bus_routes[(s, d)])
            car_prob, bus_prob = inst.transport_probs(
                s, d, solver.cb_dist, solver.c0_dist, solver.b0_dist,
                solver.last_xs,
                flows=flows, car_latency=car_lat, bus_latency=bus_lat,
            )
            demand = inst.demands[(s, d)]
            
            congestion += car_prob * demand * (car_lat - base_car_lat)
            congestion += bus_prob * demand * (bus_lat - base_bus_lat)
            base_latency += car_prob * demand * base_car_lat
            base_latency += bus_prob * demand * base_bus_lat
        
        return np.round(congestion / base_latency, 4)


class BusRidership(Objective):
    NAME = "Bus Usage"
    
    def __init__(self):
        super().__init__(self.NAME)
    
    def collect(self, solver: ScipySolver) -> float:
        if solver.last_xs is None:
            raise ValueError("Solver has no vars.")
        inst = solver.inst
        flows = inst.get_edge_flows(solver.last_xs)
        
        bus_users = 0.0
        travellers = 0.0
        for (s, d) in inst.source_dest_pairs:
            car_lat = min(inst.get_car_latencies(s, d, flows))
            bus_lat = min(inst.get_bus_latencies(s, d, flows))
            car_prob, bus_prob = inst.transport_probs(
                s, d, solver.cb_dist, solver.c0_dist, solver.b0_dist,
                solver.last_xs,
                flows=flows, car_latency=car_lat, bus_latency=bus_lat,
            )
            
            bus_users += bus_prob * inst.demands[(s, d)]
            travellers += (bus_prob + car_prob) * inst.demands[(s, d)]
        
        return np.round(bus_users / travellers, 4)


class JourniesMade(Objective):
    NAME = "Travellers"
    
    def __init__(self):
        super().__init__(self.NAME)
    
    def collect(self, solver: ScipySolver) -> float:
        if solver.last_xs is None:
            raise ValueError("Solver has no vars.")
        inst = solver.inst
        flows = inst.get_edge_flows(solver.last_xs)
        
        journies_made = 0.0
        total_demand = 0.0
        for (s, d) in inst.source_dest_pairs:
            car_lat = min(inst.get_car_latencies(s, d, flows))
            bus_lat = min(inst.get_bus_latencies(s, d, flows))
            car_prob, bus_prob = inst.transport_probs(
                s, d, solver.cb_dist, solver.c0_dist, solver.b0_dist,
                solver.last_xs,
                flows=flows, car_latency=car_lat, bus_latency=bus_lat,
            )
            
            journies_made += (car_prob + bus_prob) * inst.demands[(s, d)]
            total_demand += inst.demands[(s, d)]
        
        return np.round(journies_made / total_demand, 4)
