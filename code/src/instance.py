from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple, Union
import numpy as np
from scipy.stats import norm


from src.graph import *


Demands = Dict[SourceDestPair, float]
NormalDist = Tuple[float, float]


ALPHA_CAR = "alpha_car"
ALPHA_BUS = "alpha_bus"
ZETA_CAR = "zeta_car"
ZETA_BUS = "zeta_bus"
BETA_CAR = "beta_car"
BETA_BUS = "beta_bus"
ZETA_NONE = "zeta_none"


def default_latency(base, flow):
    return base + flow**2


# We assume the car and bus graphs are the same
class Instance:
    def __init__(
        self,
        car_graph: AdjacencyGraph,
        bus_graph: AdjacencyGraph,
        demands: Demands,
        variables: dict,
        n_paths: Union[None, int] = None,
        override = False,
        latency_fn = default_latency,
    ):
        self.car_graph = car_graph
        self.bus_graph = bus_graph
        self.demands = demands
        self.variables = variables
        self.latency_fn = latency_fn

        # Find (s,d) pairs
        self.source_dest_pairs = [sd for sd in self.demands.keys()]
        self.pair_ids = {(s, d): i for (i, (s, d)) in enumerate(self.source_dest_pairs)}
        
        # Find all paths
        car_paths: List[IdPath] = []
        self.car_routes: Dict[SourceDestPair, List[IdPath]] = {}
        self.bus_routes: Dict[SourceDestPair, List[Path]] = {}
        for sd in self.source_dest_pairs:
            self.car_routes[sd] = []
            self.bus_routes[sd] = []
        
        if override: # OVERRIDES WILL RETURN AT THIS POINT ==============================
            return
        
        total_routes = 0
        path_id = 0
        for (s, d) in self.source_dest_pairs:
            if n_paths:
                new_car_paths = self.car_graph.get_n_shortest_paths(s, d, n_paths)
                new_bus_paths = self.bus_graph.get_n_shortest_paths(s, d, n_paths)
            else:
                new_car_paths = self.car_graph.get_all_paths(s, d)
                new_bus_paths = self.bus_graph.get_all_paths(s, d)
            car_id_paths = []
            for path in new_car_paths:
                car_id_paths.append(IdPath(path, path_id))
                path_id += 1
            self.car_routes[(s, d)] = car_id_paths
            self.bus_routes[(s, d)] = new_bus_paths
            total_routes += len(new_car_paths)
            car_paths += car_id_paths

        self.total_routes = total_routes
        
        # Create a map from an edge to all paths containing that edge
        self.paths_with_edge: Dict[Edge, Set[IdPath]] = {
            edge: set() for edge in self.car_graph.edges()
        }
        for path in car_paths:
            for edge in path.edge_set:
                self.paths_with_edge[edge].add(path)
    
    def get_edge_flows(self, xs) -> Dict[Edge, float]:
        flows = {}
        for edge in self.car_graph.edges():
            flow = 0
            for path in self.paths_with_edge[edge]:
                if xs[path.id] == 0:
                    continue
                flow += xs[path.id] * self.demands[(path.start, path.end)]
            flows[edge] = flow
        return flows

    def get_car_latencies(self, source, dest, flows) -> List[float]:
        latencies = []
        for path in self.car_routes[(source, dest)]:
            latencies.append(
                # sum(self.car_graph.edge_len(edge) + flows[edge]**2
                sum(self.latency_fn(self.car_graph.edge_len(edge), flows[edge])
                    for edge in path.edge_set))
        return latencies
    
    
    def get_bus_latencies(self, source, dest, flows):
        latencies = []
        for path in self.bus_routes[(source, dest)]:
            total = 0
            for edge in path.edge_set:
                if edge in self.car_graph.edges():
                    total += self.latency_fn(self.bus_graph.edge_len(edge), flows[edge])
                else:
                    total += self.latency_fn(self.bus_graph.edge_len(edge), 0)
            latencies.append(total)
        return latencies
    
    def transport_probs(
        self,
        s: Node,
        d: Node,
        cb_dist: NormalDist,
        c0_dist: NormalDist,
        b0_dist: NormalDist,
        xs=None,
        flows=None,
        car_latency=None,
        bus_latency=None,
    ) -> Tuple[float, float]:
        if flows is None:
            flows = self.get_edge_flows(xs)
        if car_latency is None:
            car_latency = min(self.get_car_latencies(s, d, flows))
        if bus_latency is None:
            bus_latency = min(self.get_bus_latencies(s, d, flows))
        vs = self.variables
        x_cb = vs[BETA_CAR] * car_latency - vs[BETA_BUS] * bus_latency
        x_c0 = vs[BETA_CAR] * car_latency
        x_b0 = vs[BETA_BUS] * bus_latency
        pr_cb = 1 - norm.cdf(x_cb, loc=cb_dist[0], scale=cb_dist[1])
        pr_c0 = 1 - norm.cdf(x_c0, loc=c0_dist[0], scale=c0_dist[1])
        pr_b0 = 1 - norm.cdf(x_b0, loc=b0_dist[0], scale=b0_dist[1])
        prob_car = np.float64(pr_cb * pr_c0)
        prob_bus = np.float64((1 - pr_cb) * pr_b0)
        return prob_car, prob_bus


@dataclass
class InstanceParams:
    car_graph: AdjacencyGraph
    bus_graph: AdjacencyGraph
    variables: Dict
    n_paths: Union[None, int]
    demand_fn: Union[Callable[[], float], None]
    demands: Union[Demands, None]
    
    def __repr__(self):
        values = [self.car_graph, self.bus_graph, self.variables]
        string = "Instance(\n\t"
        string += "\n\t".join(map(lambda s: repr(s), values))
        string += "\n)"
        return string

    def copy(self):
        return deepcopy(self)


def new_instance(params: InstanceParams) -> Instance:
    if params.demands:
        demands = params.demands
    elif params.demand_fn:
        demands = generate_demands(params.car_graph, params.demand_fn)
    else:
        raise ValueError("InstanceParams must include either demands or demand_fn.")
    instance = Instance(
        params.car_graph,
        params.bus_graph,
        demands,
        params.variables,
        params.n_paths,
    )
    return instance


def generate_demands(graph: AdjacencyGraph, rand_var: Callable[[], float]) -> Demands:
    demands = {}
    for source in graph.nodes():
        for dest in graph.nodes():
            if source == dest:
                continue
            demands[(source, dest)] = rand_var()
    return demands
