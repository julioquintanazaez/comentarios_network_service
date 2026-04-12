import networkx as nx
from itertools import combinations
from text_utils import get_word_set

def build_comment_network(comments_data):
    """
    Construye red de comentarios basada en palabras compartidas
    
    Args:
        comments_data: Lista de diccionarios con keys:
            - id_comentario (str o int)
            - texto (str)
            - sentimiento (str: pos/neg/neu)
            - peso (float)
            - probabilidades (dict: {'pos': float, 'neg': float, 'neu': float})
    
    Returns:
        networkx.Graph
    """
    G = nx.Graph()
    
    # Preprocesar todos los comentarios
    tokens_cache = {}
    for comment in comments_data:
        cid = str(comment['id_comentario'])
        texto = comment['texto']
        tokens_cache[cid] = get_word_set(texto)
        
        # Añadir nodo con atributos
        G.add_node(cid,
                   texto=texto,
                   sentimiento=comment['sentimiento'],
                   peso=comment['peso'],
                   probabilidades=comment['probabilidades'])
    
    # Crear enlaces entre comentarios
    comment_ids = list(tokens_cache.keys())
    for i, j in combinations(range(len(comment_ids)), 2):
        cid1 = comment_ids[i]
        cid2 = comment_ids[j]
        
        shared_words = tokens_cache[cid1].intersection(tokens_cache[cid2])
        
        if shared_words:
            # Peso = número de palabras compartidas
            weight = len(shared_words)
            G.add_edge(cid1, cid2, 
                      weight=weight,
                      shared_words=list(shared_words))
    
    return G