from src.graph import *
from src.instance import *



class PathInstance(Instance):
    """
    An implementation of Instance that allows a limited set of paths to be used.
    Initially includes the shortest paths.
    
    NB: Very strongly coupled to Instance.
    """
    def __init__(
        self,
        car_graph: AdjacencyGraph,
        bus_graph: AdjacencyGraph,
        demands: Demands,
        variables: dict,
        max_paths = 80,
        latency_fn = default_latency,
    ):
        super().__init__(car_graph, bus_graph, demands, variables,
                         override=True, latency_fn=latency_fn)
        
        self.car_routes: Dict[SourceDestPair, List[IdPath]] = {
            sd: [] for sd in self.source_dest_pairs
        }
        
        self.rel_path_flows: List[float] = []
        self.active_car_paths: List[IdPath] = []
        self.active_car_paths_set: Dict[Tuple, IdPath] = {}
        self.active_bus_paths: Set[Tuple] = set()
        
        self.paths_with_edge: Dict[Edge, Set[IdPath]] = {}
        self.update_paths_with_edge([])
        
    def add_new_bus_path(self, path: Path) -> bool:
        tup = path.to_tuple()
        if tup in self.active_bus_paths:
            return False
        self.active_bus_paths.add(tup)
        sd = path.source_dest_pair()
        self.bus_routes[sd].append(path)
        return True
    
    def add_new_car_path(self, path: Path) -> Union[IdPath, None]:
        tup = path.to_tuple()
        if tup in self.active_car_paths_set:
            return None
        new_id = len(self.rel_path_flows)
        new_id_path = IdPath(path, new_id)
        sd = new_id_path.source_dest_pair()
        
        self.rel_path_flows.append(0.0)
        self.active_car_paths_set[tup] = new_id_path
        self.active_car_paths.append(new_id_path)
        self.car_routes[sd].append(new_id_path)
        
        return new_id_path
    
    def update_paths_with_edge(self, active_car_paths: List[IdPath]):
        self.paths_with_edge = {
            edge: set() for edge in self.car_graph.edges()
        }
        for path in active_car_paths:
            for edge in path.edge_set:
                self.paths_with_edge[edge].add(path)
        # return self.paths_with_edge
    
    def edge_lats(self, flows: Dict[Edge, float]):
        lats = flows.copy()
        for e, f in flows.items():
            l = self.car_graph.edge_len(e)
            lats[e] = self.latency_fn(l, f)
        return lats

    def lowest_lat_route(self, flows, paths: List[IdPath]) -> Tuple[Path, float]:
        latencies = []
        for path in paths:
            lat = sum(self.latency_fn(self.car_graph.edge_len(edge), flows[edge])
                      for edge in path.edge_set)
            latencies.append((path, lat))
        return min(latencies, key=lambda x: x[1])
    
    
    
    
@dataclass
class PathInstanceParams(InstanceParams):
    max_paths: int
    latency_fn: Callable[[float, float], float] = default_latency


def new_path_instance(params: PathInstanceParams) -> PathInstance:
    if params.demands:
        demands = params.demands
    elif params.demand_fn:
        demands = generate_demands(params.car_graph, params.demand_fn)
    else:
        raise ValueError("PathInstanceParams must include either demands or demand_fn.")
    instance = PathInstance(
        params.car_graph,
        params.bus_graph,
        demands,
        params.variables,
        params.max_paths,
        params.latency_fn,
    )
    return instance
