from heapq import heappop, heappush
import math
from typing import Dict, List, Set, Tuple, Union
from copy import deepcopy


Node = str
Edge = Tuple[Node, Node]
SourceDestPair = Tuple[Node, Node]


class GraphError(Exception):
    pass

class PathError(GraphError):
    pass


# Edge Utils ###################################################################

def flip_edge(edge: Edge):
    return (edge[1], edge[0])

def undirected(edge: Edge) -> Edge:
    if edge[0] < edge[1]:
        return (edge[0], edge[1])
    return (edge[1], edge[0])

def apply_directed(edge: Edge, directed: bool) -> Edge:
    if directed:
        return edge
    else:
        return undirected(edge)
    
################################################################################

class Path:
    def __init__(self, nodes: List[Node], length):
        if len(nodes) == 0:
            raise PathError("Path cannot be empty.")
        self.start = nodes[0]
        self.end = nodes[-1]
        self.node_set = set(nodes)
        self.node_list = nodes.copy()
        self.length = length
        self.edge_set: Set[Edge] = set()
        for i in range(len(nodes) - 1):
            self.edge_set.add((nodes[i], nodes[i + 1]))
    
    def append_node(self, new: Node, edge_len):
        self.node_set.add(new)
        self.node_list.append(new)
        self.edge_set.add((self.end, new))
        self.end = new
        self.length += edge_len
    
    def copy(self):
        return deepcopy(self)
    
    def contains_edge(self, edge: Edge):
        return edge in self.edge_set

    def contains_node(self, node: Node):
        return node in self.node_set

    def source_dest_pair(self) -> SourceDestPair:
        return (self.start, self.end)
    
    def __repr__(self) -> str:
        nodes = "->".join(self.node_list)
        return "Path(" + nodes + ")"

    def to_tuple(self):
        return tuple(self.node_list)


class IdPath(Path):
    def __init__(self, path: Path, id: int):
        super().__init__(path.node_list, path.length)
        self.id = id

################################################################################

class AdjacencyGraph:    
    def __init__(self, adjacencies: Dict[Node, List[Tuple[Node, float]]]):
        self.adjacencies = deepcopy(adjacencies)
        self.edge_lengths: Dict[Edge, float] = {}
        for start in self.adjacencies.keys():
            for (end, length) in self.adjacencies[start]:
                self.edge_lengths[(start, end)] = length
    
    def is_equal(self, other):
        for node in self.adjacencies.keys():
            if node not in other.adjacencies:
                return False
            if len(self.adjacencies[node]) != len(other.adjacencies[node]):
                return False
            if set(self.adjacencies[node]) != set(other.adjacencies[node]):
                return False
        for edge in self.edge_lengths.keys():
            if edge not in other.edge_lengths:
                return False
            if self.edge_lengths[edge] != self.edge_lengths[edge]:
                return False
        return True
        
    def nodes(self):
        return self.adjacencies.keys()

    def edges(self):
        return self.edge_lengths.keys()
    
    def edge_len(self, edge: Edge):
        return self.edge_lengths[edge]
    
    def remove_edge(self, edge: Edge):
        self.edge_lengths.pop(edge)
        self.adjacencies[edge[0]] = [
            (n, l) for (n, l) in self.adjacencies[edge[0]] if n != edge[1]
        ]

    def get_n_shortest_paths(self, start: Node, end: Node, n: int) -> List[Path]:
        if n == 0:
            return []
        paths = self.get_all_paths(start, end)
        return sorted(paths, key=lambda p: p.length)[:n]
    
    # Get all paths between two nodes via a DFS
    def get_all_paths(self, start: Node, end: Node) -> List[Path]:
        paths = []
        self._get_all_paths(start, end, Path([start], 0), paths)
        return paths
    
    def _get_all_paths(self, start: Node, end: Node, cur_path: Path, paths: List[Path]):
        if start == end:
            paths.append(cur_path.copy())
            return
        for (node, length) in self.adjacencies[start]:
            if cur_path.contains_node(node): # This is inefficient, but so is DFS, so we'll leave it for now
                continue
            new_path = cur_path.copy()
            new_path.append_node(node, length)
            self._get_all_paths(node, end, new_path, paths)

    def get_shortest_path(self, start: Node, end: Node) -> Path:
        if start not in self.adjacencies or end not in self.adjacencies:
            raise GraphError("Start or end node not in graph.")
        
        pq = [(0.0, start, [])]
        shortest_dist = {node: math.inf for node in self.adjacencies}
        shortest_dist[start] = 0

        while pq:
            cur_len, cur_node, cur_path = heappop(pq)

            if cur_node == end:
                length = self.edge_lengths[(cur_path[-1], end)]
                return Path(cur_path + [end], cur_len + length)

            if cur_len > shortest_dist[cur_node]:
                continue
            
            for neighbor, length in self.adjacencies[cur_node]:
                new_len = cur_len + length
                if new_len < shortest_dist[neighbor]:
                    shortest_dist[neighbor] = new_len
                    heappush(pq, (new_len, neighbor, cur_path + [cur_node]))

        raise GraphError(f"No path found between nodes {start}->{end}.")

    def get_fastest_path(self, sd: SourceDestPair, lengths: Dict[Edge, float]):
        """Gets the fastest path between the start and end based on the edge lengths provided."""
        start, end = sd
        if start not in self.adjacencies or end not in self.adjacencies:
            raise GraphError("Start or end node not in graph.")

        pq = [(0.0, start, [])]
        shortest_dist = {node: math.inf for node in self.adjacencies}
        shortest_dist[start] = 0

        while pq:
            cur_len, cur_node, cur_path = heappop(pq)

            if cur_node == end:
                # cur_edge = (cur_path[-1], end)
                # length = lengths.get(cur_edge, self.edge_len(cur_edge))
                return Path(cur_path + [end], cur_len)

            if cur_len > shortest_dist[cur_node]:
                continue
            
            for neighbor, _ in self.adjacencies[cur_node]:
                cur_edge = (cur_node, neighbor)
                new_len = cur_len + lengths.get(cur_edge, self.edge_len(cur_edge))
                if new_len < shortest_dist[neighbor]:
                    shortest_dist[neighbor] = new_len
                    heappush(pq, (new_len, neighbor, cur_path + [cur_node]))

        raise GraphError(f"No path found between nodes {start}->{end}.")
    
    def copy(self):
        return deepcopy(self)

################################################################################
 

def build_simple_graph(adjacencies: List[Tuple[Node, List[Node]]], edge_len: float):
    graph_adj = {}
    for (start, adj) in adjacencies:
        graph_adj[start] = [(a, edge_len) for a in adj]
    return AdjacencyGraph(graph_adj)


ADJ_NINE_GRID = "adj_nine_grid"
_ADJ_NINE_GRID = build_simple_graph([
    ("A1", ["A2", "B1"]),
    ("A2", ["A1", "A3", "B2"]),
    ("A3", ["A2", "B3"]),
    ("B1", ["A1", "B2", "C1"]),
    ("B2", ["A2", "B1", "B3", "C2"]),
    ("B3", ["A3", "B2", "C3"]),
    ("C1", ["B1", "C2"]),
    ("C2", ["B2", "C1", "C3"]),
    ("C3", ["B3", "C2"])
], 1.0)

QUAD_GRID = "quad_grid"
_QUAD_GRID = build_simple_graph([
    ("A1", ["A2", "B1", "B2"]),
    ("A2", ["A1", "B2"]),
    ("B1", ["A1", "B2"]),
    ("B2", ["A1", "A2", "B1"]),
], 1.0)

BASIC_CITY = "basic_city"
_BASIC_CITY = AdjacencyGraph({
    'L1': [('L2', 2.0), ('M1', 2.0), ('R1', 2.0)],
    'L2': [('L1', 2.0), ('M2', 1.0), ('R2', 1.0)],
    'M1': [('L1', 2.0), ('M2', 1.0), ('R1', 2.0)],
    'M2': [('L2', 1.0), ('M1', 1.0), ('R2', 1.0)],
    'R1': [('L1', 2.0), ('M1', 2.0), ('R2', 1.0)],
    'R2': [('L2', 1.0), ('M2', 1.0), ('R1', 1.0)],
})

_GRAPHS = {
    ADJ_NINE_GRID: _ADJ_NINE_GRID,
    QUAD_GRID: _QUAD_GRID,
    BASIC_CITY: _BASIC_CITY,
}

def get_graph(graph_name):
    return _GRAPHS[graph_name].copy()

def test():
    graph = get_graph(QUAD_GRID)
    print(graph.get_shortest_path("A1", "A2"))
    print(graph.get_shortest_path("B2", "B1"))
    graph_2 = graph.copy()
    graph_2.remove_edge(("A1", "A2"))
    p = graph_2.get_shortest_path("A1", "A2")
    print(p, p.length)
    p = graph_2.get_shortest_path("B2", "B1")
    print(p, p.length)
    graph_3 = graph.copy()
    graph_3.remove_edge(("B2", "B1"))
    print(graph_3.get_shortest_path("A1", "A2"))
    print(graph_3.get_shortest_path("B2", "B1"))

def test_2():
    print()
    graph = get_graph(BASIC_CITY)
    ps = graph.get_all_paths("L1", "R2")
    for p in ps:
        print(p, p.length)
    print()
    ps = graph.get_n_shortest_paths("L1", "R2", 7)
    for p in ps:
        print(p, p.length)
    

if __name__ == "__main__":
    test()
    test_2()
