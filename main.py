from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import json
import numpy as np
from typing import List, Dict
import networkx as nx
from datetime import datetime


from network_builder import build_comment_network
from community_analyzer import (
    detect_communities, 
    classify_community_sentiment,
    find_community_outliers,
    get_community_centers
)
from bridge_detector import detect_bridges_between_communities
from metrics import network_strength, community_strength, inter_community_distance
from models import NetworkAnalysisOutput, BridgeOutput, CommunityOutput


app = FastAPI(
    title="Comment Network Analysis Service API",
    description="Analiza redes de comentarios basadas en similitud léxica",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "service": "Comment Network Analysis Service API",
        "version": "1.0.0",
        "description": "Analiza redes de comentarios basadas en similitud léxica",
        "endpoints": {
            "POST /analyze": "Upload CSV or JSON file for analysis",
            "GET /health": "Health check"
        },
        "input_formats": {
            "CSV": "File with columns: id_comentario, text, sentimiento, peso, probabilidades",
            "JSON": "Structure: {'data': [{'id_comentario': 1, 'text': 'Some comment', 'sentimiento': 'pos', 'peso': '1', 'probabilidades': '10.5'}]}"
        },
        "example_usage": {
            "csv and json": "curl -X POST -F 'file=@sales.csv' http://host/analyze",
        }
    }


@app.post("/run_comments_analysis", response_model=NetworkAnalysisOutput)
async def analyze_comment_network(
    file: UploadFile = File(..., description="JSON o CSV con comentarios"),
    resolution: float = 1.0,
    outlier_percentile: int = 10
):
    """
    Analiza red de comentarios y devuelve comunidades, puentes y métricas
    
    Formato esperado del archivo:
    - id_comentario (str/int)
    - texto (str)
    - sentimiento (pos/neg/neu)
    - peso (float)
    - probabilidades (JSON string o dict en JSON)
    """
    
    # Leer archivo
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            data = json.loads(content)
            if isinstance(data, dict) and 'comentarios' in data:
                data = data['comentarios']
        elif file.filename.endswith('.csv'):
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            # Convertir probabilidades si es string JSON
            if 'probabilidades' in df.columns and df['probabilidades'].dtype == 'object':
                df['probabilidades'] = df['probabilidades'].apply(json.loads)
            data = df.to_dict('records')
        else:
            raise HTTPException(400, "Formato no soportado. Use JSON o CSV")
        
        # Validar campos mínimos
        required_fields = ['id_comentario', 'texto', 'sentimiento', 'peso', 'probabilidades']
        for record in data:
            if not all(field in record for field in required_fields):
                raise HTTPException(400, f"Falta campo requerido. Necesita: {required_fields}")
        
        # Construir red
        G = build_comment_network(data)
        
        if G.number_of_nodes() == 0:
            return JSONResponse({
                "error": "No se pudo construir la red",
                "num_comments": 0
            }, status_code=400)
        
        # Detectar comunidades
        partition, communities = detect_communities(G, resolution=resolution)
        
        if not communities:
            return JSONResponse({
                "error": "No se detectaron comunidades",
                "num_comments": G.number_of_nodes()
            }, status_code=400)
        
        # Sentimiento de comunidades
        comm_sentiment = classify_community_sentiment(communities, G)
        
        # Outliers
        outliers = find_community_outliers(G, partition, communities, outlier_percentile)
        
        # Centros de comunidades
        centers = get_community_centers(G, communities)
        
        # Puentes entre comunidades (TU FORMATO PREFERIDO)
        bridges = detect_bridges_between_communities(G, partition)
        
        # Fuerza global
        global_strength = network_strength(G)
        
        # Fuerza por comunidad
        community_strengths = {}
        for comm_id, nodes in communities.items():
            community_strengths[comm_id] = community_strength(G, partition, comm_id, nodes)
        
        # Construir respuesta
        communities_output = {}
        for comm_id, nodes in communities.items():
            communities_output[comm_id] = CommunityOutput(
                sentiment=comm_sentiment[comm_id],
                strength=community_strengths[comm_id],
                members=nodes,
                outliers=outliers.get(comm_id, []),
                center_comment_id=centers.get(comm_id)
            )
        
        # Convertir puentes al modelo
        bridges_output = [
            BridgeOutput(
                community_A=b['community_A'],
                community_B=b['community_B'],
                bridging_comment_ids=b['bridging_comment_ids'],
                total_connections=b['total_connections'],
                avg_weight=b['avg_weight']
            ) for b in bridges
        ]
        
        return NetworkAnalysisOutput(
            num_comments=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            global_strength=global_strength,
            communities=communities_output,
            bridges_between_communities=bridges_output
        )
        
    except json.JSONDecodeError:
        raise HTTPException(400, "Error al decodificar JSON")
    except Exception as e:
        raise HTTPException(500, f"Error interno: {str(e)}")
    

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "method": "analyze_comment_network",
        "supported_formats": ["CSV", "JSON"]
    }


# ==================== MAIN (for local development) ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)