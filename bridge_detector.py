from collections import defaultdict, Counter

def detect_bridges_between_communities(G, partition):
    """
    Detecta comentarios que conectan diferentes comunidades
    
    Returns:
        Lista de diccionarios con formato:
        {
            "community_A": int,
            "community_B": int,
            "bridging_comment_ids": [id1, id2, ...],
            "total_connections": int,
            "avg_weight": float
        }
    """
    # Diccionario para almacenar conexiones entre comunidades
    # Key: (comA, comB) con comA < comB
    connections = defaultdict(lambda: {
        'bridging_comments': set(),
        'edge_weights': []
    })
    
    # Analizar cada arista
    for u, v, data in G.edges(data=True):
        comm_u = partition[u]
        comm_v = partition[v]
        
        if comm_u != comm_v:
            # Ordenar comunidades para tener clave única
            com_a, com_b = sorted([comm_u, comm_v])
            weight = data.get('weight', 1)
            
            # Añadir comentarios puente
            connections[(com_a, com_b)]['bridging_comments'].add(u)
            connections[(com_a, com_b)]['bridging_comments'].add(v)
            connections[(com_a, com_b)]['edge_weights'].append(weight)
    
    # Convertir a formato de salida
    bridges = []
    for (com_a, com_b), data in connections.items():
        weights = data['edge_weights']
        avg_weight = sum(weights) / len(weights) if weights else 0
        
        bridges.append({
            "community_A": com_a,
            "community_B": com_b,
            "bridging_comment_ids": list(data['bridging_comments']),
            "total_connections": len(weights),
            "avg_weight": round(avg_weight, 3)
        })
    
    # Ordenar por total_connections descendente (puentes más fuertes primero)
    bridges.sort(key=lambda x: x['total_connections'], reverse=True)
    
    return bridges