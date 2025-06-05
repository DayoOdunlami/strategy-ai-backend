# models.py - Pydantic Models for API Requests/Responses (WITH FEEDBACK SYSTEM)
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
    enhanced_features: Optional[Dict[str, Any]] = None  # For multi-agent responses
    chat_log_id: Optional[str] = None  # For feedback linking

# ============================================================================
# USER FEEDBACK MODELS (NEW)
# ============================================================================

class UserFeedback(BaseModel):
    chat_log_id: Optional[str] = None
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5, description="1-5 star rating")
    feedback_type: str = Field(default="general", description="response_quality, source_relevance, general")
    comment: Optional[str] = Field(None, max_length=1000)
    helpful: Optional[bool] = None  # Quick thumbs up/down

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
# REPORT GENERATION MODELS
# ============================================================================

class ReportGenerationRequest(BaseModel):
    report_type: str  # strategy_analysis, project_similarity, etc.
    sector: str = "General"
    format: str = "pdf"  # pdf, docx, both
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

class ReportDownloadResponse(BaseModel):
    success: bool
    report_id: str
    download_urls: List[str]
    metadata: ReportMetadata
    message: str

class ReportListResponse(BaseModel):
    reports: List[ReportMetadata]
    total_count: int

# ============================================================================
# MAP DATA MODELS
# ============================================================================

class MapRegion(BaseModel):
    region_id: str
    name: str
    code: str
    description: str
    color: str
    major_cities: List[str]

class MapStation(BaseModel):
    name: str
    city: str
    region: str
    station_type: str  # terminus, interchange, etc.
    coordinates: List[float]  # [lng, lat]

class MapProject(BaseModel):
    id: str
    name: str
    description: str
    status: str
    location: str
    coordinates: List[float]
    documents_count: int
    completion: int

class RouteInformation(BaseModel):
    start_station: str
    end_station: str
    distance_km: float
    estimated_time_minutes: int
    intermediate_stations: List[str]
    route_geometry: Dict[str, Any]

class LocationSearchResult(BaseModel):
    name: str
    type: str  # station, city, region
    region: str
    coordinates: List[float]
    relevance: float

class MapConfiguration(BaseModel):
    default_center: List[float]
    default_zoom: int
    max_zoom: int
    min_zoom: int
    tile_layers: Dict[str, str]
    style_config: Dict[str, Any]
    interaction: Dict[str, Any]

# ============================================================================
# AI AGENTS MODELS
# ============================================================================

class AgentInfo(BaseModel):
    name: str
    specialization: str
    status: str
    last_tested: datetime

class AgentRequest(BaseModel):
    type: str  # chat, analysis, report, metadata, comprehensive
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

# ============================================================================
# ENHANCED ANALYTICS MODELS (Including Feedback)
# ============================================================================

class AgentStatusResponse(BaseModel):
    agents: List[AgentInfo]
    orchestrator_status: str
    total_agents: int

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

# ============================================================================
# CUSTOM REPORT TEMPLATES
# ============================================================================

class ReportTemplate(BaseModel):
    name: str
    title: str
    sections: List[str]
    description: str
    parameters: Dict[str, Any] = {}

class CustomTemplateRequest(BaseModel):
    template_name: str
    title: str
    sections: List[str]
    description: Optional[str] = ""
    parameters: Dict[str, Any] = {}

class TemplateListResponse(BaseModel):
    templates: List[Dict[str, Any]]
    total_count: int

class UseCaseSuggestion(BaseModel):
    suggested_use_case: str
    confidence: float
    description: str
    example_queries: List[str] = []
    explanation: str

# ============================================================================
# DOCUMENT MANAGEMENT MODELS
# ============================================================================

class DocumentUpload(BaseModel):
    title: Optional[str] = None
    sector: str = "General"
    use_case: Optional[str] = None
    tags: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = {}

class URLScrapeRequest(BaseModel):
    url: str
    depth: str = "single"  # single, comprehensive
    max_pages: int = 10
    follow_nav_links: bool = False
    include_downloads: bool = False
    sector: str = "General"
    use_case: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    filename: Optional[str]
    sector: str
    use_case: Optional[str]
    tags: Optional[str]
    source_type: str  # file, url
    source_url: Optional[str]
    status: str  # processing, completed, failed
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    feedback_count: Optional[int] = 0  # Number of feedback entries
    average_rating: Optional[float] = None  # Average user rating

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool

# ============================================================================
# METADATA MANAGEMENT MODELS
# ============================================================================

class MetadataUpdate(BaseModel):
    title: Optional[str] = None
    sector: Optional[str] = None
    use_case: Optional[str] = None
    tags: Optional[str] = None
    topic: Optional[str] = None
    custom_fields: Optional[Dict[str, str]] = {}

class BulkMetadataUpdate(BaseModel):
    document_ids: List[str]
    updates: MetadataUpdate

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    title: Optional[str]
    topic: Optional[str]
    tags: Optional[str]
    sector: str
    use_case: Optional[str]
    confidence: float = 0.8
    metadata: Dict[str, Any] = {}

# ============================================================================
# SECTOR & USE CASE MODELS
# ============================================================================

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

# ============================================================================
# CSV IMPORT/EXPORT MODELS
# ============================================================================

class CSVImportRequest(BaseModel):
    update_existing: bool = True
    validate_only: bool = False

class CSVImportResponse(BaseModel):
    success: bool
    import_id: str
    records_to_process: int
    message: str
    status_endpoint: str

class ImportStatus(BaseModel):
    import_id: str
    status: str  # processing, completed, failed
    total_records: int
    processed_records: int = 0
    updated_count: int = 0
    created_count: int = 0
    failed_count: int = 0
    errors: List[str] = []
    started_at: datetime
    completed_at: Optional[datetime] = None

# ============================================================================
# BACKGROUND TASK STATUS MODELS
# ============================================================================

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class DocumentProcessingStatus(ProcessingStatus):
    document_id: str
    filename: str
    chunks_created: int = 0
    estimated_completion: Optional[datetime] = None

class ScrapingStatus(ProcessingStatus):
    url: str
    pages_scraped: int = 0
    max_pages: int
    current_page: Optional[str] = None

# ============================================================================
# SEARCH & FILTER MODELS
# ============================================================================

class SearchFilter(BaseModel):
    sector: Optional[str] = None
    use_case: Optional[str] = None
    source_type: Optional[str] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search_text: Optional[str] = None
    min_rating: Optional[float] = None  # Filter by minimum user rating

class SearchResult(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    relevance_score: float
    metadata: Dict[str, Any]
    highlighted_text: Optional[str] = None
    user_rating: Optional[float] = None  # Average user rating for this document

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    filters_applied: SearchFilter
    suggestions: List[str] = []

# ============================================================================
# ANALYTICS & ADMIN MODELS (Enhanced with Feedback)
# ============================================================================

class ActivityLog(BaseModel):
    id: str
    action: str  # upload, delete, update, chat, feedback, etc.
    entity_type: str  # document, sector, use_case, etc.
    entity_id: Optional[str]
    user_type: str
    details: Dict[str, Any]
    timestamp: datetime

class UserActivity(BaseModel):
    session_id: str
    user_type: str
    queries_count: int
    sectors_used: List[str]
    use_cases_used: List[str]
    feedback_given: int  # Number of feedback entries
    average_rating_given: Optional[float]  # Average rating this user gives
    last_activity: datetime
    session_duration: Optional[int] = None  # seconds

# ============================================================================
# ERROR & VALIDATION MODELS
# ============================================================================

class APIError(BaseModel):
    error: str
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ValidationError(BaseModel):
    field: str
    message: str
    invalid_value: Any

class BulkOperationResult(BaseModel):
    success: bool
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[ValidationError] = []
    message: str

# ============================================================================
# CONFIGURATION & SETTINGS MODELS
# ============================================================================

class SystemSettings(BaseModel):
    max_file_size_mb: int = 50
    allowed_file_types: List[str] = [".pdf", ".docx", ".txt", ".csv", ".md"]
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    max_chunks_per_document: int = 1000
    vector_search_top_k: int = 8
    ai_model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-ada-002"
    enable_user_feedback: bool = True  # New setting for feedback system

class AdminSettings(BaseModel):
    auto_tag_documents: bool = True
    require_sector_assignment: bool = True
    allow_csv_import: bool = True
    max_scraping_pages: int = 50
    analytics_retention_days: int = 365
    feedback_moderation_enabled: bool = False  # Auto-moderate feedback content

# ============================================================================
# SYSTEM ANALYTICS MODELS
# ============================================================================

class SystemMetric(BaseModel):
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    status: str = "normal"  # normal, warning, critical

class PerformanceMetric(BaseModel):
    endpoint: str
    response_time: float
    error_rate: float
    requests_per_minute: float
    timestamp: datetime
    status: str = "normal"

class SystemHealthResponse(BaseModel):
    status: str
    health_score: float
    metrics: List[SystemMetric]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime

class PerformanceAnalysisResponse(BaseModel):
    overall_status: str
    performance_score: float
    metrics: List[PerformanceMetric]
    bottlenecks: List[Dict[str, Any]]
    optimizations: List[str]
    timestamp: datetime

class FeedbackAnalysisResponse(BaseModel):
    satisfaction_score: float
    feedback_count: int
    sentiment_analysis: Dict[str, float]
    key_themes: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime

class ComprehensiveAnalysisResponse(BaseModel):
    system_health: SystemHealthResponse
    performance: PerformanceAnalysisResponse
    feedback: Optional[FeedbackAnalysisResponse]
    overall_status: str
    priority_actions: List[str]
    timestamp: datetime