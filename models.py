from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum

class Sentiment(str, Enum):
    pos = "pos"
    neg = "neg"
    neu = "neu"

class CommentInput(BaseModel):
    comment_id: str
    text: str
    sentiment: Sentiment
    weight: float
    probabilities: Dict[str, float]

class BridgeOutput(BaseModel):
    community_A: int
    community_B: int
    bridging_comment_ids: List[str]
    total_connections: int
    avg_weight: float

class CommunityOutput(BaseModel):
    sentiment: str
    strength: float
    members: List[str]
    outliers: List[str]
    center_comment_id: Optional[str]

class NetworkAnalysisOutput(BaseModel):
    num_comments: int
    num_edges: int
    global_strength: float
    communities: Dict[int, CommunityOutput]
    bridges_between_communities: List[BridgeOutput]