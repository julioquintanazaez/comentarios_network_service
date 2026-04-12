import community as community_louvain
import networkx as nx
from collections import defaultdict

def detect_communities(G, resolution=1.0):
    """
    Detecta comunidades usando algoritmo Louvain
    
    Args:
        G: networkx Graph
        resolution: parámetro de resolución para Louvain (mayor = más comunidades)
    
    Returns:
        partition: dict {node: community_id}
        communities: dict {community_id: [nodes]}
    """
    if G.number_of_nodes() == 0:
        return {}, {}
    
    # Ajustar resolución si es necesario
    partition = community_louvain.best_partition(G, 
                                                 weight='weight',
                                                 resolution=resolution)
    
    # Agrupar por comunidad
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    return partition, dict(communities)

def classify_community_sentiment(communities, G):
    """
    Determina sentimiento dominante de cada comunidad basado en pesos y probabilidades
    """
    comm_sentiment = {}
    
    for comm_id, nodes in communities.items():
        if not nodes:
            comm_sentiment[comm_id] = 'neu'
            continue
        
        # Acumular probabilidades ponderadas
        pos_sum = 0.0
        neg_sum = 0.0
        neu_sum = 0.0
        total_weight = 0.0
        
        for node in nodes:
            peso = G.nodes[node].get('peso', 1.0)
            probs = G.nodes[node].get('probabilidades', {'pos': 0.33, 'neg': 0.33, 'neu': 0.34})
            
            pos_sum += probs.get('pos', 0) * peso
            neg_sum += probs.get('neg', 0) * peso
            neu_sum += probs.get('neu', 0) * peso
            total_weight += peso
        
        if total_weight > 0:
            avg_pos = pos_sum / total_weight
            avg_neg = neg_sum / total_weight
            avg_neu = neu_sum / total_weight
            
            sentiment_scores = [('pos', avg_pos), ('neg', avg_neg), ('neu', avg_neu)]
            dominant = max(sentiment_scores, key=lambda x: x[1])[0]
        else:
            dominant = 'neu'
        
        comm_sentiment[comm_id] = dominant
    
    return comm_sentiment

def find_community_outliers(G, partition, communities, percentile_threshold=10):
    """
    Detecta outliers dentro de cada comunidad usando closeness centrality
    
    Args:
        percentile_threshold: percentil inferior para considerar outlier (ej. 10 = 10% más bajo)
    """
    outliers = {}
    
    for comm_id, nodes in communities.items():
        if len(nodes) < 4:  # Comunidades muy pequeñas no tienen outliers significativos
            outliers[comm_id] = []
            continue
        
        # Subgrafo de la comunidad
        subgraph = G.subgraph(nodes)
        if subgraph.number_of_nodes() < 3:
            outliers[comm_id] = []
            continue
        
        # Calcular closeness centrality
        try:
            closeness = nx.closeness_centrality(subgraph, distance='weight')
            if closeness:
                # Usar percentil en lugar de umbral fijo
                closeness_values = list(closeness.values())
                threshold = np.percentile(closeness_values, percentile_threshold)
                comm_outliers = [node for node in nodes if closeness.get(node, 0) <= threshold]
                outliers[comm_id] = comm_outliers
            else:
                outliers[comm_id] = []
        except:
            outliers[comm_id] = []
    
    return outliers

def get_community_centers(G, communities):
    """
    Encuentra el comentario más central en cada comunidad (mayor betweenness centrality)
    """
    centers = {}
    
    for comm_id, nodes in communities.items():
        if len(nodes) < 2:
            centers[comm_id] = nodes[0] if nodes else None
            continue
        
        subgraph = G.subgraph(nodes)
        try:
            betweenness = nx.betweenness_centrality(subgraph, weight='weight')
            if betweenness:
                center = max(betweenness, key=betweenness.get)
                centers[comm_id] = center
            else:
                centers[comm_id] = nodes[0] if nodes else None
        except:
            centers[comm_id] = nodes[0] if nodes else None
    
    return centers