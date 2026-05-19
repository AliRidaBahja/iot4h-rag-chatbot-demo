from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    GPT3_5_TURBO = "gpt-3.5-turbo"

class QueryInput(BaseModel):
    query: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O)

class QueryResponse(BaseModel):
    response: Optional[str] = None 
    session_id: str
    model: ModelName 
    sources: Optional[List[Dict]] = None
    is_done: bool = False
    static_qa: Optional[Dict[str, Any]] = None
