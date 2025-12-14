from pprint import pprint
import numpy as np
import math

from src.graph import *
from src.instance import *
from src.pathinstance import PathInstance


CarUsage = Dict[SourceDestPair, Tuple[float, float]]


class ColgenSolver:
    """
    Uses a bi-level column generation / path swapping algorithm
    to find a user equilibrium in the given transport network.
    Algorithm adapted from Lu et al (2009).\n
    https://doi.org/10.1016/j.trb.2008.07.005
    """
    def __init__(self, instance: PathInstance):
        self.inst = instance
        self.last_result = None
        self.last_xs = None

    def solve(self, verbose=True,
              max_outer=10, max_inner=200, min_gap=1.0, min_improve=0.01,
              init_car_usage=0.5, gap_threshold=0.01,
    ):
        if verbose:
            print("Solving with Colgen solver...")
        vs = self.inst.variables
        mu_cb = vs[ZETA_CAR][0] - vs[ZETA_BUS][0] + vs[ALPHA_BUS] - vs[ALPHA_CAR]
        mu_c0 = vs[ZETA_CAR][0] - vs[ZETA_NONE][0] - vs[ALPHA_CAR]
        mu_b0 = vs[ZETA_BUS][0] - vs[ZETA_NONE][0] - vs[ALPHA_BUS]
        sigma_cb = ( vs[ZETA_CAR][1]**2 + vs[ZETA_BUS][1]**2 ) ** (1/2)
        sigma_c0 = ( vs[ZETA_CAR][1]**2 + vs[ZETA_NONE][1]**2 ) ** (1/2)
        sigma_b0 = ( vs[ZETA_BUS][1]**2 + vs[ZETA_NONE][1]**2 ) ** (1/2)
        self.cb_dist = (mu_cb, sigma_cb)
        self.c0_dist = (mu_c0, sigma_c0)
        self.b0_dist = (mu_b0, sigma_b0)

        # PathInstance automatically intialises paths (1)
                
        car_usage = {sd: (init_car_usage, init_car_usage) for sd in self.inst.source_dest_pairs}
        
        # Find the shortest paths
        for (s, d) in self.inst.source_dest_pairs:
            shortest_car = self.inst.car_graph.get_shortest_path(s, d)
            path = self.inst.add_new_car_path(shortest_car)
            if path:
                self.inst.rel_path_flows[path.id] = 1.0
            else:
                raise ValueError(f"Couldn't find initial path from {s}->{d}.")
            shortest_bus = self.inst.bus_graph.get_shortest_path(s, d)
            self.inst.add_new_bus_path(shortest_bus)
        self.inst.update_paths_with_edge(self.inst.active_car_paths)
        
        abs_path_flows = self.get_abs_flows(self.inst.rel_path_flows, car_usage)
        flows = self.inst.get_edge_flows(abs_path_flows)
        total_flow_gap, total_modal_gap, car_latencies = self.get_gaps(flows, car_usage, True)
        total_gap = total_flow_gap + total_modal_gap
        
        # Outer Loop
        for outer_loops in range(max_outer):
            if verbose:
                print(f"Outer Loop Iter: {outer_loops}")
                print(f"Current Gap: {total_gap}")
            # (2) Add new paths
            new_paths = 0
            for sd in self.inst.source_dest_pairs:
                # new_shortest, lat = self.inst.lowest_lat_route(flows, self.inst.possible_car_routes[sd])
                edge_lats = self.inst.edge_lats(flows)
                new_shortest = self.inst.car_graph.get_fastest_path(sd, edge_lats)
                new_id_path = self.inst.add_new_car_path(new_shortest)
                if new_id_path:
                    car_latencies[new_id_path.id] = (new_shortest.length, new_shortest.length)
                    for path in self.inst.car_routes[sd]:
                        cur_lat, _ = car_latencies[path.id]
                        car_latencies[path.id] = (cur_lat, new_shortest.length)
                    # self.inst.add_active_path(sd, new_id_path)
                    new_paths += 1
                new_shortest_bus = self.inst.bus_graph.get_fastest_path(sd, edge_lats)
                self.inst.add_new_bus_path(new_shortest_bus)
            self.inst.update_paths_with_edge(self.inst.active_car_paths)
            if new_paths == 0:
                abs_path_flows = self.get_abs_flows(self.inst.rel_path_flows, car_usage)
                flows = self.inst.get_edge_flows(abs_path_flows)
                total_flow_gap, total_modal_gap, car_latencies = self.get_gaps(flows, car_usage, True)
            
            if verbose:
                print(f"Added {new_paths} new paths")
            
            # Inner Loop
            # (5) set l and prev_gap
            prev_gap = 0
            for inner_loops in range(max_inner):
                # (6) Descent direction
                rho = (1.0 / (outer_loops + 1)) if inner_loops == 0 else 1.0
                if inner_loops > int(inner_loops / 2):
                    rho = (1.0 / (inner_loops - int(inner_loops / 2)))
                # rho = (1.0 / (inner_loops + 1))
                self.swap_modes(car_usage, rho)
                self.swap_paths(self.inst.rel_path_flows, car_latencies, rho)
                # (7) Load network / (8) Calc shortest paths
                abs_path_flows = self.get_abs_flows(self.inst.rel_path_flows, car_usage)
                flows = self.inst.get_edge_flows(abs_path_flows)
                total_flow_gap, total_modal_gap, car_latencies = self.get_gaps(flows, car_usage, False)
                # (9) Update objective function
                total_gap = total_flow_gap + total_modal_gap
                # (10) Check convergence
                if abs(total_gap - prev_gap) <= min_improve * rho:
                    if verbose:
                        print("Terminating Inner Loop: Min improve achieved")
                    break
                prev_gap = total_gap
            if verbose:
                print(f"Inner loops: {inner_loops}")
            
            # (4) Check convergence
            total_flow_gap, total_modal_gap, car_latencies = self.get_gaps(flows, car_usage, True)
            total_gap = total_flow_gap + total_modal_gap
            gap_minimised = total_gap <= min_gap
            no_new_paths = new_paths == 0
            if gap_minimised and no_new_paths:
                print("Terminating Outer: Gap minimised and no new paths.")
                break
            gap_below_threshold = total_gap < gap_threshold
            if gap_below_threshold:
                print("Terminating Outer: Gap minimised below threshold.")
                break
        if verbose and not (gap_minimised and no_new_paths) and not gap_below_threshold:
            print("Terminating Outer Loop: iteration limit reached.")
        if verbose:
            total_flow_gap, total_modal_gap, car_latencies = self.get_gaps(flows, car_usage, True)
            print(f"Final gap: {total_gap} ({total_flow_gap} + {total_modal_gap})")
        
        self.last_xs = self.flow_fractions(self.inst.rel_path_flows, car_usage)
        return self.last_xs
    
    def get_abs_flows(self, rel_flows: List[float], car_usage: CarUsage) -> List[float]:
        abs_flows = rel_flows.copy()
        for i in range(len(self.inst.active_car_paths)):
            sd = self.inst.active_car_paths[i].source_dest_pair()
            abs_flows[i] *= self.inst.demands[sd] * car_usage[sd][0]
        return abs_flows
    
    def flow_fractions(self, rel_flows: List[float], car_usage: CarUsage) -> List[float]:
        effective_flows = rel_flows.copy()
        for i in range(len(self.inst.active_car_paths)):
            sd = self.inst.active_car_paths[i].source_dest_pair()
            effective_flows[i] *= car_usage[sd][0]
        return effective_flows
    
    def swap_modes(self, car_usage: CarUsage, rho: float):
        for sd in self.inst.source_dest_pairs:
            current, target = car_usage[sd]
            delta = rho * (current - target)
            car_usage[sd] = (current - delta, target)
    
    def swap_paths(self, rel_flows: List[float], car_latencies: Dict[int, Tuple[float, float]], rho: float):
        for sd in self.inst.source_dest_pairs:
            ids = [path.id for path in self.inst.car_routes[sd]]
            zero_gap_ids = []
            total_delta = 0
            for id in ids:
                lat, min_lat = car_latencies[id]
                gap = lat - min_lat
                if gap > 0:
                    delta_x = min(rel_flows[id], rel_flows[id] * rho * (lat - min_lat) / lat)
                    total_delta += delta_x
                    rel_flows[id] = rel_flows[id] - delta_x
                else:
                    zero_gap_ids.append(id)
            # if len(zero_gap_ids) == 0:
            #     print(car_latencies)
            #     raise ValueError("No paths with 0 gap, this shouldn't be possible.")
            for id in zero_gap_ids:
                rel_flows[id] += total_delta / len(zero_gap_ids)
    
    def get_gaps(self, flows: Dict[Edge, float], car_usage: CarUsage, consider_all: bool, debug=False):
        total_flow_gap = 0
        n_gaps = 0
        total_modal_gap = 0
        car_latencies: Dict[int, Tuple[float, float]] = {}
        
        edge_lats = self.inst.edge_lats(flows)
        
        for (s, d) in self.inst.source_dest_pairs:
            
            car_lats = self.inst.get_car_latencies(s, d, flows)
            if consider_all:  # Also consider paths not currently in our model
                min_car_lat = min(car_lats)
                shortest_path = self.inst.car_graph.get_fastest_path((s, d), edge_lats)
                min_car_lat = min(min_car_lat, shortest_path.length)
            else:
                min_car_lat = min(car_lats)
            for (lat, path) in zip(car_lats, self.inst.car_routes[(s, d)]):
                weight = self.inst.rel_path_flows[path.id] * self.inst.demands[(s, d)] * car_usage[(s, d)][0]
                gap = (lat - min_car_lat) * weight
                car_latencies[path.id] = (lat, min_car_lat)
                if debug and gap > 0.1:
                    print(lat, min_car_lat, shortest_path)
                total_flow_gap += gap
                if weight != 0:
                    n_gaps += 1
            
            bus_lats = self.inst.get_bus_latencies(s, d, flows)
            if len(bus_lats) > 0:
                min_bus_lat = min(bus_lats)
            else:
                min_bus_lat = math.inf
            car_prob, bus_prob = self.inst.transport_probs(
                s, d, self.cb_dist, self.c0_dist, self.b0_dist,
                flows=flows, car_latency=min_car_lat, bus_latency=min_bus_lat, 
            )
            # We only need to consider car - others can be calculated directly
            # as they don't affect the equilibrium
            cur_usage = car_usage[(s, d)][0]
            car_usage[(s, d)] = (cur_usage, car_prob)
            cur_model_gap = abs(cur_usage - car_prob)
            total_modal_gap += cur_model_gap
        
        total_flow_gap /= max(n_gaps, 1)
        
        return (total_flow_gap, total_modal_gap, car_latencies)

def main():
    # demand_distr = lambda: max(random.normalvariate(100, 10), 10)
    # demand_distr = lambda: max(np.random.normal(4.0, 1.0), 0.1)
    demand_distr = lambda: 4
    # adj = get_graph(QUAD_GRID)
    adj = get_graph(ADJ_NINE_GRID)
    
    demands = generate_demands(adj, demand_distr)
    
    variables = {
        ALPHA_CAR: 1.0,
        BETA_CAR: 1.0,
        ZETA_CAR: (1.0, 1.0),
        ALPHA_BUS: 1.0,
        BETA_BUS: 1.0,
        ZETA_BUS: (1.0, 1.0),
        ZETA_NONE: (-1000.0, 1.0),
    }
    
    inst = PathInstance(adj, adj, demands, variables)
    solver = ColgenSolver(inst)
    solution = solver.solve(verbose=True)
    vars = np.round(solution, 2)
    # pprint(vars)
    for path in inst.active_car_paths:
        print(vars[path.id], path.node_list, sep="\t")



if __name__ == "__main__":
    main()

