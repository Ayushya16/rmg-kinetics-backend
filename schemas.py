from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictByFeatures(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    A: float
    n: float
    Ea_kJ_per_mol: float
    model_used: str
    meta: Optional[Dict[str, Any]] = None
