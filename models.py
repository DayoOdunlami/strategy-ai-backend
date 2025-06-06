# models.py - Essential Pydantic Models for Strategy AI Backend
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============================================================================
# CHAT & QUERY MODELS
# ============================================================================

class ChatMessage(BaseModel):
    message: str
    sector: Optional[str] = "General"
    use_case: Optional[str] = None
    session_id: Optional[str] = None
    user_type: str = "public"  # public, admin, analyst

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.8
    suggested_use_case: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    enhanced_features: Optional[Dict[str, Any]] = None

# ============================================================================
# USER FEEDBACK MODELS
# ============================================================================

class UserFeedback(BaseModel):
    chat_log_id: Optional[str] = None
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5, description="1-5 star rating")
    feedback_type: str = Field(default="general", description="response_quality, source_relevance, general")
    comment: Optional[str] = Field(None, max_length=1000)
    helpful: Optional[bool] = None

class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: str
    message: str

# ============================================================================
# DOCUMENT MODELS
# ============================================================================

class DocumentUpload(BaseModel):
    title: Optional[str] = None
    sector: str = "General"
    use_case: Optional[str] = None
    tags: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = {}

class DocumentResponse(BaseModel):
    id: str
    title: str
    filename: Optional[str]
    sector: str
    use_case: Optional[str]
    tags: Optional[str]
    source_type: str  # file, url
    status: str  # processing, completed, failed
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int
    limit: int = 10
    offset: int = 0
    has_more: bool = False

# ============================================================================
# SYSTEM MODELS
# ============================================================================

class SystemAnalytics(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    total_sectors: int = 0
    total_use_cases: int = 0
    documents_by_sector: Dict[str, int] = {}
    documents_by_use_case: Dict[str, int] = {}
    recent_activity_count: int = 0
    storage_usage: Dict[str, Any] = {}

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# ============================================================================
# SEARCH MODELS
# ============================================================================

class SearchFilter(BaseModel):
    sector: Optional[str] = None
    use_case: Optional[str] = None
    source_type: Optional[str] = None
    tags: Optional[List[str]] = None
    search_text: Optional[str] = None

class SearchResult(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    relevance_score: float
    metadata: Dict[str, Any]
    highlighted_text: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    filters_applied: SearchFilter
    suggestions: List[str] = []

# ============================================================================
# ADMIN MODELS
# ============================================================================

class APIError(BaseModel):
    error: str
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class SystemSettings(BaseModel):
    max_file_size_mb: int = 50
    allowed_file_types: List[str] = [".pdf", ".docx", ".txt", ".csv", ".md"]
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    max_chunks_per_document: int = 1000
    vector_search_top_k: int = 8
    ai_model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-ada-002"
    enable_user_feedback: bool = True 