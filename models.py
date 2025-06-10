# models.py - Pydantic Models for Multi-Agent Strategy AI System
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
    model: Optional[str] = None  # For model selection

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.8
    suggested_use_case: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    enhanced_features: Optional[Dict[str, Any]] = None
    chat_log_id: Optional[str] = None
    model_used: Optional[str] = None
    agents_used: Optional[List[str]] = None

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

class FeedbackAnalytics(BaseModel):
    total_feedback: int
    average_rating: float
    helpful_percentage: float
    recent_feedback: List[Dict[str, Any]]
    rating_distribution: Dict[str, int]
    feedback_by_type: Dict[str, int]
    feedback_trends: Dict[str, Any]

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
    source_type: str
    source_url: Optional[str]
    status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    feedback_count: Optional[int] = 0
    average_rating: Optional[float] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool

# ============================================================================
# SEARCH MODELS
# ============================================================================

class SearchFilter(BaseModel):
    sector: Optional[str] = None
    use_case: Optional[str] = None
    source_type: Optional[str] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search_text: Optional[str] = None
    min_rating: Optional[float] = None

class SearchResult(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    relevance_score: float
    metadata: Dict[str, Any]
    highlighted_text: Optional[str] = None
    user_rating: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    filters_applied: SearchFilter
    suggestions: List[str] = []

# ============================================================================
# AI AGENTS MODELS
# ============================================================================

class AgentInfo(BaseModel):
    name: str
    specialization: str
    status: str
    last_tested: datetime

class AgentRequest(BaseModel):
    type: str
    query: str
    context: str
    sector: str = "General"
    use_case: Optional[str] = None
    user_type: str = "public"
    complexity: str = "simple"
    parameters: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    agent: str
    response: str
    confidence: float
    response_type: str
    metadata: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None

class OrchestrationResponse(BaseModel):
    primary_response: str
    agents_used: List[str]
    response_type: str
    detailed_results: Optional[Dict[str, Any]] = None
    confidence: float

class AgentStatusResponse(BaseModel):
    agents: List[AgentInfo]
    orchestrator_status: str
    total_agents: int

# ============================================================================
# ANALYTICS MODELS
# ============================================================================

class SystemAnalytics(BaseModel):
    total_documents: int
    total_chunks: int
    total_sectors: int
    total_use_cases: int
    documents_by_sector: Dict[str, int]
    documents_by_use_case: Dict[str, int]
    documents_by_source_type: Dict[str, int]
    recent_activity_count: int
    storage_usage: Dict[str, Any]

class EnhancedSystemAnalytics(SystemAnalytics):
    agent_performance: Dict[str, Any] = {}
    report_generation_stats: Dict[str, Any] = {}
    map_data_usage: Dict[str, Any] = {}
    advanced_features_usage: Dict[str, Any] = {}
    feedback_summary: FeedbackAnalytics

class SystemMetric(BaseModel):
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    status: str = "normal"

class SystemHealthResponse(BaseModel):
    status: str
    health_score: float
    metrics: List[SystemMetric]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime

# ============================================================================
# REPORT GENERATION MODELS
# ============================================================================

class ReportGenerationRequest(BaseModel):
    report_type: str
    sector: str = "General"
    format: str = "pdf"
    title: Optional[str] = None
    scope: str = "comprehensive"
    parameters: Dict[str, Any] = {}

class ReportMetadata(BaseModel):
    report_id: str
    report_type: str
    generated_at: datetime
    parameters: Dict[str, Any]
    files: List[Dict[str, str]]
    word_count: int
    sections_count: int
    ai_agents_used: List[str]

# ============================================================================
# MISC MODELS
# ============================================================================

class UseCaseSuggestion(BaseModel):
    suggested_use_case: str
    confidence: float
    description: str
    example_queries: List[str] = []
    explanation: str

class MetadataUpdate(BaseModel):
    title: Optional[str] = None
    sector: Optional[str] = None
    use_case: Optional[str] = None
    tags: Optional[str] = None
    topic: Optional[str] = None
    custom_fields: Optional[Dict[str, str]] = {}

class SectorCreate(BaseModel):
    name: str
    description: str

class SectorResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    use_case_count: int = 0

class UseCaseCreate(BaseModel):
    name: str
    sector: str
    tags: Optional[str] = ""
    prompt_template: Optional[str] = ""

class UseCaseResponse(BaseModel):
    id: str
    name: str
    sector: str
    tags: Optional[str]
    prompt_template: Optional[str]
    created_at: datetime
    updated_at: datetime

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class APIError(BaseModel):
    error: str
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now) 