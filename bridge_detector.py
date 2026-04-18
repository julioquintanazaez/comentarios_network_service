from collections import defaultdict, Counter

def detect_bridges_between_communities(graph, partition):
    """
    Detects comments that connect different communities
    
    Returns:
        List of dictionaries with format:
        {
            "community_A": int,
            "community_B": int,
            "bridging_comment_ids": [id1, id2, ...],
            "total_connections": int,
            "avg_weight": float
        }
    """
    # Dictionary to store connections between communities
    # Key: (commA, commB) with commA < commB
    connections = defaultdict(lambda: {
        'bridging_comments': set(),
        'edge_weights': []
    })
    
    # Analyze each edge
    for node_u, node_v, edge_data in graph.edges(data=True):
        community_u = partition[node_u]
        community_v = partition[node_v]
        
        if community_u != community_v:
            # Sort communities to have unique key
            community_a, community_b = sorted([community_u, community_v])
            weight = edge_data.get('weight', 1)
            
            # Add bridging comments
            connections[(community_a, community_b)]['bridging_comments'].add(node_u)
            connections[(community_a, community_b)]['bridging_comments'].add(node_v)
            connections[(community_a, community_b)]['edge_weights'].append(weight)
    
    # Convert to output format
    bridges = []
    for (community_a, community_b), connection_data in connections.items():
        weights = connection_data['edge_weights']
        avg_weight = sum(weights) / len(weights) if weights else 0
        
        bridges.append({
            "community_A": community_a,
            "community_B": community_b,
            "bridging_comment_ids": list(connection_data['bridging_comments']),
            "total_connections": len(weights),
            "avg_weight": round(avg_weight, 3)
        })
    
    # Sort by total_connections descending (strongest bridges first)
    bridges.sort(key=lambda bridge: bridge['total_connections'], reverse=True)
    
    return bridges