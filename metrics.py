import networkx as nx
import numpy as np

def network_strength(G):
    """Fuerza global de la red = densidad * peso promedio"""
    if G.number_of_edges() == 0:
        return 0.0
    
    density = nx.density(G)
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    avg_weight = np.mean(weights) if weights else 0
    
    return round(density * avg_weight, 4)

def community_strength(G, partition, comm_id, nodes):
    """Fuerza de una comunidad individual"""
    if len(nodes) < 2 or G.number_of_edges() == 0:
        return 0.0
    
    subgraph = G.subgraph(nodes)
    if subgraph.number_of_edges() == 0:
        return 0.0
    
    # Densidad interna
    internal_density = nx.density(subgraph)
    
    # Peso promedio interno
    internal_weights = [data.get('weight', 1) for _, _, data in subgraph.edges(data=True)]
    avg_internal_weight = np.mean(internal_weights) if internal_weights else 0
    
    # Conectividad externa (normalizada)
    external_edges = 0
    for node in nodes:
        for neighbor in G.neighbors(node):
            if partition[neighbor] != comm_id:
                external_edges += 1
    
    max_external = len(nodes) * (G.number_of_nodes() - len(nodes))
    external_norm = external_edges / max_external if max_external > 0 else 0
    
    # Fuerza = densidad_interna * peso_interno / (1 + conectividad_externa)
    strength = (internal_density * avg_internal_weight) / (1 + external_norm)
    
    return round(strength, 4)

def inter_community_distance(G, partition, com_a, com_b, nodes_a, nodes_b):
    """Distancia entre comunidades (menor = más conectadas)"""
    if not nodes_a or not nodes_b:
        return 1.0
    
    # Contar conexiones entre comunidades
    connections = 0
    max_possible = len(nodes_a) * len(nodes_b)
    
    for u in nodes_a:
        for v in nodes_b:
            if G.has_edge(u, v):
                connections += G[u][v].get('weight', 1)
    
    if max_possible == 0:
        return 1.0
    
    connectivity = connections / max_possible
    distance = 1 - connectivity
    
    return round(distance, 4)