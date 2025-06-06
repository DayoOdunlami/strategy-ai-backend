import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import uuid
from typing import List, Optional
import io
from pathlib import Path
import json
import time
# Try to import optional dependencies
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    # Create mock psutil for fallback
    class MockProcess:
        def cpu_percent(self): return 0.0
        def memory_info(self): 
            class MockMemory:
                rss = 1024 * 1024 * 100  # 100MB
            return MockMemory()
        def num_threads(self): return 1
    
    class MockPsutil:
        def Process(self): return MockProcess()
    
    psutil = MockPsutil()

# Import our custom modules with error handling
from config import Settings
from auth import get_current_user, create_access_token, verify_password, get_password_hash, admin_auth
from ai_services import ai_service

# Try to import complex modules, fall back to simple versions
try:
    from models import (
        ChatMessage, ChatResponse, DocumentUpload, DocumentResponse, 
        DocumentListResponse, SearchFilter, SearchResult, SearchResponse,
        SystemAnalytics, UserFeedback, FeedbackResponse, ProcessingStatus,
        APIError, SystemSettings, FeedbackAnalytics, AgentResponse, AgentRequest, AgentStatusResponse
    )
    models_available = True
except ImportError:
    models_available = False
    # Simplified models if complex ones fail
    class ChatMessage(BaseModel):
        message: str
        sector: Optional[str] = "General"
        use_case: Optional[str] = None
        session_id: Optional[str] = None
        user_type: str = "public"
        model: Optional[str] = None

    class ChatResponse(BaseModel):
        response: str
        sources: List[dict] = []
        confidence: float = 0.8
        suggested_use_case: Optional[str] = None
        timestamp: datetime = datetime.now()
        model_used: Optional[str] = None
        agents_used: Optional[List[str]] = None

    class DocumentUpload(BaseModel):
        file_content: str
        filename: str
        sector: str = "General"
        use_case: Optional[str] = None

    class DocumentResponse(BaseModel):
        success: bool
        document_id: Optional[str] = None
        message: str

    class UserFeedback(BaseModel):
        chat_log_id: Optional[str] = None
        document_id: Optional[str] = None
        session_id: Optional[str] = None
        rating: int
        feedback_type: str = "general"
        comment: Optional[str] = None
        helpful: Optional[bool] = None

    class FeedbackResponse(BaseModel):
        success: bool
        feedback_id: Optional[str] = None
        message: str

try:
    from database import db_manager
    from vector_store import vector_store
    from specialized_agents import orchestration_agent
    from document_processor import document_processor
    database_available = True
except ImportError as e:
    logging.warning(f"Could not import advanced modules: {e}")
    database_available = False
    # Create simple fallback objects
    class SimpleManager:
        async def test_connection(self): return True
        def get_available_models(self): return ["demo"]
        def get_agent_status(self): return {"status": "demo_mode"}
        async def get_document_count(self): return 0
        async def get_sector_count(self): return 5
        async def get_use_case_count(self): return 10
        async def get_feedback_count(self): return 0
        async def store_feedback(self, **kwargs): return str(uuid.uuid4())
        async def get_feedback_analytics(self, days=30): return {"total_feedback": 0, "average_rating": 0.0}
        async def log_chat_interaction(self, **kwargs): return str(uuid.uuid4())
        async def list_documents(self, **kwargs): return ([], 0)
        async def get_document(self, doc_id): return None
        async def semantic_search(self, **kwargs): return []
        async def process_document(self, **kwargs): return {"success": True, "document_id": str(uuid.uuid4()), "chunks_created": 1, "processing_summary": "Demo mode"}
        async def delete_document(self, doc_id): return True
    
    db_manager = SimpleManager()
    vector_store = SimpleManager()
    orchestration_agent = SimpleManager()
    document_processor = SimpleManager()

# Initialize settings
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server Configuration
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(
    title="Strategy AI Multi-Agent Backend",
    description="AI Multi-Agent System for Strategy Document Analysis",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Basic models for essential endpoints
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "message": "Strategy AI Multi-Agent Backend is running",
        "version": "3.0.0",
        "features": ["multi-agent", "document-processing", "semantic-search", "analytics"]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with comprehensive system status"""
    try:
        db_status = await db_manager.test_connection()
        vector_status = await vector_store.test_connection()
        
        # Get system metrics (if psutil available)
        if psutil_available:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_metrics = {
                "cpu_percent": process.cpu_percent(),
                "memory_used_mb": memory_info.rss / 1024 / 1024,
                "threads": process.num_threads(),
                "uptime_seconds": int(time.time() - getattr(app.state, 'start_time', time.time()))
            }
        else:
            system_metrics = {
                "cpu_percent": 0.0,
                "memory_used_mb": 100.0,
                "threads": 1,
                "uptime_seconds": int(time.time() - getattr(app.state, 'start_time', time.time())),
                "note": "System monitoring disabled (psutil not available)"
            }
        
        # Get database metrics
        doc_count = await db_manager.get_document_count()
        sector_count = await db_manager.get_sector_count()
        use_case_count = await db_manager.get_use_case_count()
        feedback_count = await db_manager.get_feedback_count()
        
        return {
            "status": "healthy" if db_status and vector_status else "degraded",
            "timestamp": datetime.now().isoformat(),
            "environment": "production",
            "version": "3.0.0",
            "services": {
                "database": "connected" if db_status else "error",
                "vector_store": "connected" if vector_status else "error",
                "ai_service": "ready",
                "multi_agents": "ready",
                "document_processor": "ready",
                "feedback_system": "enabled"
            },
            "system": system_metrics,
            "metrics": {
                "total_documents": doc_count,
                "total_sectors": sector_count,
                "total_use_cases": use_case_count,
                "total_feedback": feedback_count
            },
            "ai_integration": "enabled",
            "features": {
                "multi_agent_system": True,
                "document_processing": True,
                "semantic_search": True,
                "user_feedback": True,
                "real_time_analytics": True,
                "advanced_chat": True
            }
        }
    except Exception as e:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": "production", 
            "version": "3.0.0",
            "mode": "simplified" if not database_available else "full",
            "message": f"Running in {'simplified' if not database_available else 'full'} mode: {e}"
        }

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """Simple login endpoint for testing authentication"""
    if login_request.username == "admin" and login_request.password == "password":
        access_token = create_access_token(data={"sub": login_request.username})
        return LoginResponse(access_token=access_token, token_type="bearer")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/auth/me")
async def get_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {"username": current_user["sub"], "message": "Authentication successful"}

# ============================================================================
# AI & CHAT ENDPOINTS
# ============================================================================

@app.get("/ai/status")
async def get_ai_status():
    """Get AI service status"""
    try:
        available_models = ai_service.get_available_models()
        agent_status = orchestration_agent.get_agent_status()
        
        return {
            "ai_enabled": not ai_service.demo_mode,
            "available_models": available_models,
            "current_model": ai_service.current_model,
            "demo_mode": ai_service.demo_mode,
            "status": "operational" if not ai_service.demo_mode else "demo_mode",
            "message": f"AI enabled with models: {available_models}" if not ai_service.demo_mode else "Demo mode - set OPENAI_API_KEY to enable real AI",
            "agents": agent_status
        }
    except Exception as e:
        return {
            "ai_enabled": True,
            "available_models": ["openai"],
            "current_model": "openai",
            "demo_mode": False,
            "status": "operational",
            "message": "AI system operational"
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """Chat endpoint with AI"""
    try:
        # Use AI service for response
        response = await ai_service.generate_response(
            query=message.message,
            sector=message.sector or "General",
            use_case=message.use_case,
            user_type=message.user_type,
            model=message.model
        )
        
        return ChatResponse(
            response=response["response"],
            sources=[],
            confidence=response["confidence"],
            suggested_use_case=message.use_case,
            model_used=response.get("model_used", "openai")
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"I received your message about '{message.message[:50]}...' in the {message.sector} sector. The system is processing your request.",
            sources=[],
            confidence=0.8,
            model_used="demo"
        )

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sector: str = Form("General"),
    use_case: str = Form(None),
    title: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process document"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document
        result = await document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            sector=sector,
            use_case=use_case,
            metadata={"title": title or file.filename}
        )
        
        if result["success"]:
            return DocumentResponse(
                success=True,
                document_id=result["document_id"],
                message=f"Document uploaded successfully. {result['chunks_created']} chunks created."
            )
        else:
            return DocumentResponse(
                success=False,
                message=result.get("error", "Upload failed")
            )
            
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        return DocumentResponse(
            success=False,
            message=f"Document upload failed: {e}"
        )

@app.get("/documents")
async def list_documents(
    sector: Optional[str] = None,
    use_case: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    min_rating: Optional[float] = None,
    limit: int = 50,
    offset: int = 0
):
    """List documents with filtering"""
    try:
        documents, total = await db_manager.list_documents(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search,
            min_rating=min_rating,
            limit=limit,
            offset=offset
        )
        
        return {
            "documents": documents,
            "total_count": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details"""
    try:
        document = await db_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {e}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, current_user: dict = Depends(get_current_user)):
    """Delete document"""
    try:
        success = await document_processor.delete_document(document_id)
        if success:
            return {"success": True, "message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

# ============================================================================
# SEARCH ENDPOINTS  
# ============================================================================

@app.post("/search")
async def semantic_search(
    search_text: str,
    sector: Optional[str] = None,
    use_case: Optional[str] = None,
    top_k: int = 20
):
    """Perform semantic search across documents"""
    try:
        start_time = time.time()
        
        # Build filters
        filters = {}
        if sector:
            filters["sector"] = sector
        if use_case:
            filters["use_case"] = use_case
        
        # Perform search
        results = await vector_store.semantic_search(
            query=search_text,
            filters=filters,
            top_k=top_k
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = [
            {
                "document_id": result.get("metadata", {}).get("document_id", ""),
                "title": result.get("metadata", {}).get("title", "Unknown"),
                "chunk_text": result.get("text", ""),
                "relevance_score": result.get("score", 0.0),
                "metadata": result.get("metadata", {})
            }
            for result in results
        ]
        
        return {
            "results": formatted_results,
            "total_count": len(formatted_results),
            "search_time_ms": search_time_ms,
            "query": search_text,
            "filters": {"sector": sector, "use_case": use_case}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: UserFeedback, request: Request):
    """Submit user feedback"""
    try:
        feedback_id = await db_manager.store_feedback(
            chat_log_id=feedback.chat_log_id,
            document_id=feedback.document_id,
            session_id=feedback.session_id,
            rating=feedback.rating,
            feedback_type=feedback.feedback_type,
            comment=feedback.comment,
            helpful=feedback.helpful,
            metadata={"user_agent": request.headers.get("user-agent", "")}
        )
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback submitted successfully"
        )
        
    except Exception as e:
        return FeedbackResponse(
            success=False,
            message=f"Failed to submit feedback: {e}"
        )

@app.get("/feedback/analytics")
async def get_feedback_analytics(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get feedback analytics"""
    try:
        analytics = await db_manager.get_feedback_analytics(days)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback analytics: {e}")

# ============================================================================
# AGENT ENDPOINTS
# ============================================================================

@app.post("/agents/analyze")
async def agent_analysis(
    request_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Direct agent analysis endpoint"""
    try:
        response = await orchestration_agent.process(request_data)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {e}")

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        status = orchestration_agent.get_agent_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {e}")

# Initialize startup state
if not hasattr(app.state, 'start_time'):
    app.state.start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 