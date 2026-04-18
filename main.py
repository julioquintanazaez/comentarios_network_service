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
    description="Analyzes comment networks based on lexical similarity",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "service": "Comment Network Analysis Service API",
        "version": "1.0.0",
        "description": "Analyzes comment networks based on lexical similarity",
        "endpoints": {
            "POST /analyze": "Upload CSV or JSON file for analysis",
            "GET /health": "Health check"
        },
        "input_formats": {
            "CSV": "File with columns: comment_id, text, sentiment, weight, probabilities",
            "JSON": "Structure: {'data': [{'comment_id': 1, 'text': 'Some comment', 'sentiment': 'pos', 'weight': '1', 'probabilities': '10.5'}]}"
        },
        "example_usage": {
            "csv and json": "curl -X POST -F 'file=@sales.csv' http://host/analyze",
        }
    }


@app.post("/run_comments_analysis", response_model=NetworkAnalysisOutput)
async def analyze_comment_network(
    file: UploadFile = File(..., description="JSON or CSV with comments"),
    resolution: float = 1.0,
    outlier_percentile: int = 10
):
    """
    Analyzes comment network and returns communities, bridges and metrics
    
    Expected file format:
    - comment_id (str/int)
    - text (str)
    - sentiment (pos/neg/neu)
    - weight (float)
    - probabilities (JSON string or dict in JSON)
    """
    
    # Read file
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            data = json.loads(content)
            if isinstance(data, dict) and 'comments' in data:
                data = data['comments']
        elif file.filename.endswith('.csv'):
            from io import StringIO
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            # Convert probabilities if it's a JSON string
            if 'probabilities' in df.columns and df['probabilities'].dtype == 'object':
                df['probabilities'] = df['probabilities'].apply(json.loads)
            data = df.to_dict('records')
        else:
            raise HTTPException(400, "Unsupported format. Use JSON or CSV")
        
        # Validate required fields
        required_fields = ['comment_id', 'text', 'sentiment', 'weight', 'probabilities']
        for record in data:
            if not all(field in record for field in required_fields):
                raise HTTPException(400, f"Missing required field. Needs: {required_fields}")
        
        # Build network
        graph = build_comment_network(data)
        
        if graph.number_of_nodes() == 0:
            return JSONResponse({
                "error": "Could not build the network",
                "num_comments": 0
            }, status_code=400)
        
        # Detect communities
        partition, communities = detect_communities(graph, resolution=resolution)
        
        if not communities:
            return JSONResponse({
                "error": "No communities detected",
                "num_comments": graph.number_of_nodes()
            }, status_code=400)
        
        # Community sentiment
        community_sentiment = classify_community_sentiment(communities, graph)
        
        # Outliers
        outliers = find_community_outliers(graph, partition, communities, outlier_percentile)
        
        # Community centers
        centers = get_community_centers(graph, communities)
        
        # Bridges between communities (YOUR PREFERRED FORMAT)
        bridges = detect_bridges_between_communities(graph, partition)
        
        # Global strength
        global_strength = network_strength(graph)
        
        # Strength per community
        community_strengths = {}
        for community_id, nodes in communities.items():
            community_strengths[community_id] = community_strength(graph, partition, community_id, nodes)
        
        # Build response
        communities_output = {}
        for community_id, nodes in communities.items():
            communities_output[community_id] = CommunityOutput(
                sentiment=community_sentiment[community_id],
                strength=community_strengths[community_id],
                members=nodes,
                outliers=outliers.get(community_id, []),
                center_comment_id=centers.get(community_id)
            )
        
        # Convert bridges to model
        bridges_output = [
            BridgeOutput(
                community_A=bridge['community_A'],
                community_B=bridge['community_B'],
                bridging_comment_ids=bridge['bridging_comment_ids'],
                total_connections=bridge['total_connections'],
                avg_weight=bridge['avg_weight']
            ) for bridge in bridges
        ]
        
        return NetworkAnalysisOutput(
            num_comments=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            global_strength=global_strength,
            communities=communities_output,
            bridges_between_communities=bridges_output
        )
        
    except json.JSONDecodeError:
        raise HTTPException(400, "Error decoding JSON")
    except Exception as e:
        raise HTTPException(500, f"Internal error: {str(e)}")
    

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