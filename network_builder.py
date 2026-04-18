import networkx as nx
from itertools import combinations
from text_utils import get_word_set

def build_comment_network(comments_data):
    """
    Builds a comment network based on shared words
    
    Args:
        comments_data: List of dictionaries with keys:
            - comment_id (str or int)
            - text (str)
            - sentiment (str: pos/neg/neu)
            - weight (float)
            - probabilities (dict: {'pos': float, 'neg': float, 'neu': float})
    
    Returns:
        networkx.Graph
    """
    graph = nx.Graph()
    
    # Preprocess all comments
    tokens_cache = {}
    for comment in comments_data:
        comment_id = str(comment['comment_id'])
        text = comment['text']
        tokens_cache[comment_id] = get_word_set(text)
        
        # Add node with attributes
        graph.add_node(comment_id,
                       text=text,
                       sentiment=comment['sentiment'],
                       weight=comment['weight'],
                       probabilities=comment['probabilities'])
    
    # Create edges between comments
    comment_ids = list(tokens_cache.keys())
    for i, j in combinations(range(len(comment_ids)), 2):
        comment_id_1 = comment_ids[i]
        comment_id_2 = comment_ids[j]
        
        shared_words = tokens_cache[comment_id_1].intersection(tokens_cache[comment_id_2])
        
        if shared_words:
            # Weight = number of shared words
            weight = len(shared_words)
            graph.add_edge(comment_id_1, comment_id_2, 
                          weight=weight,
                          shared_words=list(shared_words))
    
    return graph