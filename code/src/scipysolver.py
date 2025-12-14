from pprint import pprint
import numpy as np
import scipy.optimize as opt

from src.graph import *
from src.instance import *


class ScipySolver:
    def __init__(self, instance: Instance):
        self.inst = instance
        self.last_result = None
        self.last_xs = None
    
    def solve(self, verbose=True):
        if verbose:
            print("Solving with Scipy solver...")
            
        ### Build math programming problem ###
        
        bounds = opt.Bounds(lb = 0.0)
        
        start = 0
        
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
        
        def x_upper_bound(xs, start: int, end: int, s: Node, d: Node):
            car_prob, _ = self.inst.transport_probs(s, d, self.cb_dist, self.c0_dist, self.b0_dist, xs)
            return (np.sum(xs[start:end]) - car_prob) * 5

        # All routes between s and d should sum to 1
        constraints = []
        for (s, d) in self.inst.source_dest_pairs:
            end = start + len(self.inst.car_routes[(s, d)])
            constraints.append({
                "type": "eq",
                "fun": x_upper_bound,
                "args": [start, end, s, d]
            })
            start = end
        
        # 
            
        xs = np.ones(self.inst.total_routes)
        
        self.last_result = opt.minimize(
            self.objective,
            xs,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={
                "disp": verbose,
                "maxiter": 5000,
            },
        )
        self.last_xs = self.last_result.x
        
        return self.last_result

    def objective(self, xs):
        total = 0
        
        flows = self.inst.get_edge_flows(xs)
        for edge in self.inst.car_graph.edges():
            flow = flows[edge]
            # Obtained by integrating the latency function [(edge len) + (flow)^2]
            length = self.inst.car_graph.edge_len(edge)
            total += length * flow + length * flow**3 / 3
        return total / self.inst.total_routes

    def get_probs(self, s, d, xs):
        return self.inst.transport_probs(s, d, self.cb_dist, self.c0_dist, self.b0_dist, xs)


def main():
    # demand_distr = lambda: max(random.normalvariate(100, 10), 10)
    # demand_distr = lambda: np.random.normal(4.0, 1.0)
    demand_distr = lambda: 4
    # adj = get_graph(QUAD_GRID)
    adj = get_graph(BASIC_CITY)
    
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
    
    inst = Instance(adj, adj, demands, variables)
    joint = ScipySolver(inst)
    solution = joint.solve()
    vars = np.round(solution.x, 2)
    pprint(vars)



if __name__ == "__main__":
    main()
