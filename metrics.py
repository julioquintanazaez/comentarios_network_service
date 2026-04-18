import networkx as nx
import numpy as np

def network_strength(graph):
    """Global network strength = density * average weight"""
    if graph.number_of_edges() == 0:
        return 0.0
    
    density = nx.density(graph)
    weights = [edge_data.get('weight', 1) for _, _, edge_data in graph.edges(data=True)]
    avg_weight = np.mean(weights) if weights else 0
    
    return round(density * avg_weight, 4)

def community_strength(graph, partition, community_id, nodes):
    """Strength of an individual community"""
    if len(nodes) < 2 or graph.number_of_edges() == 0:
        return 0.0
    
    subgraph = graph.subgraph(nodes)
    if subgraph.number_of_edges() == 0:
        return 0.0
    
    # Internal density
    internal_density = nx.density(subgraph)
    
    # Average internal weight
    internal_weights = [edge_data.get('weight', 1) for _, _, edge_data in subgraph.edges(data=True)]
    avg_internal_weight = np.mean(internal_weights) if internal_weights else 0
    
    # External connectivity (normalized)
    external_edges = 0
    for node in nodes:
        for neighbor in graph.neighbors(node):
            if partition[neighbor] != community_id:
                external_edges += 1
    
    max_external = len(nodes) * (graph.number_of_nodes() - len(nodes))
    external_norm = external_edges / max_external if max_external > 0 else 0
    
    # Strength = internal_density * internal_weight / (1 + external_connectivity)
    strength = (internal_density * avg_internal_weight) / (1 + external_norm)
    
    return round(strength, 4)

def inter_community_distance(graph, partition, community_a, community_b, nodes_a, nodes_b):
    """Distance between communities (lower = more connected)"""
    if not nodes_a or not nodes_b:
        return 1.0
    
    # Count connections between communities
    connections = 0
    max_possible = len(nodes_a) * len(nodes_b)
    
    for node_u in nodes_a:
        for node_v in nodes_b:
            if graph.has_edge(node_u, node_v):
                connections += graph[node_u][node_v].get('weight', 1)
    
    if max_possible == 0:
        return 1.0
    
    connectivity = connections / max_possible
    distance = 1 - connectivity
    
    return round(distance, 4)