from src.graph import *

def test_build_simple_graph():
    adj = build_simple_graph([
        ("A", ["B", "C"]),
        ("B", ["A", "D", "E"]),
        ("C", ["A", "E", "F"]),
        ("D", ["B", "G"]),
        ("E", ["B", "C", "G"]),
        ("F", ["C", "G"]),
        ("G", ["D", "E", "F"]),
    ], 1.0).adjacencies
    
    expected_adj = {
        'A': [('B', 1.0), ('C', 1.0)],
        'B': [('A', 1.0), ('D', 1.0), ('E', 1.0)],
        'C': [('A', 1.0), ('E', 1.0), ('F', 1.0)],
        'D': [('B', 1.0), ('G', 1.0)],
        'E': [('B', 1.0), ('C', 1.0), ('G', 1.0)],
        'F': [('C', 1.0), ('G', 1.0)],
        'G': [('D', 1.0), ('E', 1.0), ('F', 1.0)]
    }
    
    assert adj == expected_adj


def test_get_all_paths():
    graph = build_simple_graph([
        ("A", ["B", "C"]),
        ("B", ["A", "D", "E"]),
        ("C", ["A", "E", "F"]),
        ("D", ["B", "G"]),
        ("E", ["B", "C", "G"]),
        ("F", ["C", "G"]),
        ("G", ["D", "E", "F"]),
    ], 1.0)
    
    paths = graph.get_all_paths("A", "G")
    expected_paths = [
        ['A', 'B', 'D', 'G'],
        ['A', 'B', 'E', 'C', 'F', 'G'],
        ['A', 'B', 'E', 'G'],
        ['A', 'C', 'E', 'B', 'D', 'G'],
        ['A', 'C', 'E', 'G'],
        ['A', 'C', 'F', 'G']
    ]
    for (path, ls) in zip(paths, expected_paths):
        assert path.node_list == ls

def test_get_shortest_path():
    graph = build_simple_graph([
        ("A1", ["A2", "B1", "B2"]),
        ("A2", ["A1", "B2", "B3"]),
        ("B1", ["A1", "B2", "C2"]),
        ("B2", ["A1", "A2", "B1", "B3", "C2", "C3"]),
        ("B3", ["A2", "B2", "C3"]),
        ("C2", ["B1", "B2", "C3"]),
        ("C3", ["B2", "B3", "C2"]),
    ], 1.0)
    
    expected_path = [
        "A1", "B2", "C3"
    ]
    assert expected_path == graph.get_shortest_path("A1", "C3").node_list

