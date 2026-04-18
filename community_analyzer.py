import networkx as nx
from collections import defaultdict
import numpy as np

def detect_communities(graph, resolution=1.0, seed=42):
    """
    Detects communities using NetworkX Louvain algorithm
    
    Args:
        graph: networkx Graph
        resolution: resolution parameter for Louvain (higher = more communities)
        seed: seed for reproducibility
    
    Returns:
        partition: dict {node: community_id}
        communities: dict {community_id: [nodes]}
    """
    if graph.number_of_nodes() == 0:
        return {}, {}
    
    # Detect communities using NetworkX Louvain
    communities_list = nx.community.louvain_communities(
        graph, 
        weight='weight', 
        resolution=resolution,
        seed=seed
    )
    
    # Convert to partition format {node: community_id}
    partition = {}
    for community_id, nodes in enumerate(communities_list):
        for node in nodes:
            partition[node] = community_id
    
    # Group by community (same format as before)
    communities_dict = defaultdict(list)
    for node, community_id in partition.items():
        communities_dict[community_id].append(node)
    
    return partition, dict(communities_dict)

def classify_community_sentiment(communities, graph):
    """
    Determines dominant sentiment of each community based on weights and probabilities
    """
    community_sentiment = {}
    
    for community_id, nodes in communities.items():
        if not nodes:
            community_sentiment[community_id] = 'neu'
            continue
        
        # Accumulate weighted probabilities
        pos_sum = 0.0
        neg_sum = 0.0
        neu_sum = 0.0
        total_weight = 0.0
        
        for node in nodes:
            weight = graph.nodes[node].get('weight', 1.0)
            probabilities = graph.nodes[node].get('probabilities', {'pos': 0.33, 'neg': 0.33, 'neu': 0.34})
            
            pos_sum += probabilities.get('pos', 0) * weight
            neg_sum += probabilities.get('neg', 0) * weight
            neu_sum += probabilities.get('neu', 0) * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_pos = pos_sum / total_weight
            avg_neg = neg_sum / total_weight
            avg_neu = neu_sum / total_weight
            
            sentiment_scores = [('pos', avg_pos), ('neg', avg_neg), ('neu', avg_neu)]
            dominant = max(sentiment_scores, key=lambda x: x[1])[0]
        else:
            dominant = 'neu'
        
        community_sentiment[community_id] = dominant
    
    return community_sentiment

def find_community_outliers(graph, partition, communities, percentile_threshold=10):
    """
    Detects outliers within each community using closeness centrality
    
    Args:
        percentile_threshold: lower percentile to consider as outlier (e.g., 10 = lowest 10%)
    """
    outliers = {}
    
    for community_id, nodes in communities.items():
        if len(nodes) < 4:  # Very small communities have no significant outliers
            outliers[community_id] = []
            continue
        
        # Subgraph of the community
        subgraph = graph.subgraph(nodes)
        if subgraph.number_of_nodes() < 3:
            outliers[community_id] = []
            continue
        
        # Calculate closeness centrality
        try:
            closeness = nx.closeness_centrality(subgraph, distance='weight')
            if closeness:
                # Use percentile instead of fixed threshold
                closeness_values = list(closeness.values())
                threshold = np.percentile(closeness_values, percentile_threshold)
                community_outliers = [node for node in nodes if closeness.get(node, 0) <= threshold]
                outliers[community_id] = community_outliers
            else:
                outliers[community_id] = []
        except:
            outliers[community_id] = []
    
    return outliers

def get_community_centers(graph, communities):
    """
    Finds the most central comment in each community (highest betweenness centrality)
    """
    centers = {}
    
    for community_id, nodes in communities.items():
        if len(nodes) < 2:
            centers[community_id] = nodes[0] if nodes else None
            continue
        
        subgraph = graph.subgraph(nodes)
        try:
            betweenness = nx.betweenness_centrality(subgraph, weight='weight')
            if betweenness:
                center = max(betweenness, key=betweenness.get)
                centers[community_id] = center
            else:
                centers[community_id] = nodes[0] if nodes else None
        except:
            centers[community_id] = nodes[0] if nodes else None
    
    return centers