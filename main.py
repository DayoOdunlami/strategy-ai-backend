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

# Import our custom modules
from config import Settings
from auth import get_current_user, create_access_token, verify_password, get_password_hash, admin_auth
from ai_services import ai_service
from models import (
    ChatMessage, ChatResponse, DocumentUpload, DocumentResponse, 
    DocumentListResponse, SearchFilter, SearchResult, SearchResponse,
    SystemAnalytics, UserFeedback, FeedbackResponse, ProcessingStatus,
    APIError, SystemSettings, FeedbackAnalytics, AgentResponse, AgentRequest, AgentStatusResponse
)
from database import db_manager
from vector_store import vector_store
from specialized_agents import orchestration_agent
from document_processor import document_processor

# Initialize settings
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server Configuration
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))  # Properly handle Railway's PORT

app = FastAPI(
    title="Strategy AI Multi-Agent Backend",
    description="AI Multi-Agent System for Strategy Document Analysis with Advanced Capabilities",
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

# In-memory storage (for demonstration - use database in production)
documents_store = {}
feedback_store = {}
chat_logs = {}

# Pydantic models for basic endpoints
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
    """Enhanced health check with system status"""
    try:
        db_status = await db_manager.test_connection()
        vector_status = await vector_store.test_connection()
        
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
                "document_processor": "ready"
            },
            "ai_integration": "enabled",
            "features": {
                "multi_agent_system": True,
                "document_processing": True,
                "semantic_search": True,
                "user_feedback": True,
                "real_time_analytics": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

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

@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """Example of a protected route"""
    return {
        "message": f"Hello {current_user['sub']}, this is a protected route!",
        "user": current_user,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# AI & CHAT ENDPOINTS
# ============================================================================

@app.get("/ai/status")
async def get_ai_status():
    """Get AI service and agents status"""
    available_models = ai_service.get_available_models()
    agent_status = orchestration_agent.get_agent_status()
    
    return {
        "ai_enabled": not ai_service.demo_mode,
        "available_models": available_models,
        "current_model": ai_service.current_model,
        "demo_mode": ai_service.demo_mode,
        "status": "operational" if not ai_service.demo_mode else "demo_mode",
        "message": f"AI enabled with models: {available_models}" if not ai_service.demo_mode else "Demo mode - set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable real AI",
        "agents": agent_status
    }

@app.post("/ai/model")
async def set_ai_model(model_data: dict, current_user: dict = Depends(get_current_user)):
    """Set the current AI model"""
    model = model_data.get("model")
    if ai_service.set_model(model):
        return {"success": True, "current_model": model, "message": f"Model set to {model}"}
    else:
        available = ai_service.get_available_models()
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model '{model}'. Available models: {available}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """Enhanced chat endpoint with multi-agent orchestration"""
    try:
        # Determine complexity level
        complexity = "complex" if len(message.message) > 200 or "analyze" in message.message.lower() else "simple"
        
        # Prepare agent request
        agent_request = {
            "type": "chat",
            "query": message.message,
            "sector": message.sector or "General",
            "use_case": message.use_case,
            "user_type": message.user_type,
            "complexity": complexity,
            "context": ""
        }
        
        # Use orchestration agent for processing
        if complexity == "complex":
            agent_response = await orchestration_agent.process(agent_request)
            
            response_text = agent_response.get("primary_response", agent_response.get("response", ""))
            agents_used = agent_response.get("agents_used", [])
            confidence = agent_response.get("confidence", 0.8)
            
        else:
            # Use AI service directly for simple queries
            ai_response = await ai_service.generate_response(
                query=message.message,
                sector=message.sector or "General",
                use_case=message.use_case,
                user_type=message.user_type,
                model=message.model
            )
            
            response_text = ai_response["response"]
            agents_used = []
            confidence = ai_response["confidence"]
        
        # Get relevant sources
        sources = await vector_store.semantic_search(
            query=message.message,
            filters={"sector": message.sector} if message.sector else None,
            top_k=5
        )
        
        formatted_sources = [
            {
                "document_title": source["metadata"].get("title", "Unknown"),
                "source": source["metadata"].get("source", "Unknown"),
                "relevance_score": source.get("score", 0.0),
                "chunk_preview": source["text"][:200] + "..."
            }
            for source in sources[:3]
        ]
        
        # Log interaction
        chat_log_id = await db_manager.log_chat_interaction(
            message=message.message,
            response=response_text,
            sector=message.sector or "General",
            use_case=message.use_case,
            session_id=message.session_id,
            user_type=message.user_type,
            confidence=confidence,
            sources=formatted_sources,
            agents_used=agents_used,
            model_used=message.model or ai_service.current_model
        )
        
        return ChatResponse(
            response=response_text,
            sources=formatted_sources,
            confidence=confidence,
            suggested_use_case=message.use_case,
            chat_log_id=chat_log_id,
            model_used=message.model or ai_service.current_model,
            agents_used=agents_used
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")

@app.get("/chat/history")
async def get_chat_history(session_id: Optional[str] = None, limit: int = 20):
    """Get chat history"""
    try:
        history = await db_manager.get_chat_history(session_id, limit)
        return {"history": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {e}")

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/documents/upload")
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
            return {
                "success": True,
                "document_id": result["document_id"],
                "message": f"Document uploaded successfully. {result['chunks_created']} chunks created.",
                "processing_summary": result["processing_summary"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {e}")

@app.get("/documents", response_model=DocumentListResponse)
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
        
        return DocumentListResponse(
            documents=documents,
            total_count=total,
            limit=limit,
            offset=offset,
            has_more=offset + limit < total
        )
        
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

@app.post("/search", response_model=SearchResponse)
async def semantic_search(search_filter: SearchFilter):
    """Perform semantic search across documents"""
    try:
        start_time = time.time()
        
        # Build filters
        filters = {}
        if search_filter.sector:
            filters["sector"] = search_filter.sector
        if search_filter.use_case:
            filters["use_case"] = search_filter.use_case
        
        # Perform search
        results = await vector_store.semantic_search(
            query=search_filter.search_text or "",
            filters=filters,
            top_k=20
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = [
            SearchResult(
                document_id=result["metadata"].get("document_id", ""),
                title=result["metadata"].get("title", "Unknown"),
                chunk_text=result["text"],
                relevance_score=result.get("score", 0.0),
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        return SearchResponse(
            results=formatted_results,
            total_count=len(formatted_results),
            search_time_ms=search_time_ms,
            filters_applied=search_filter
        )
        
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
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}")

@app.get("/feedback/analytics", response_model=FeedbackAnalytics)
async def get_feedback_analytics(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get feedback analytics"""
    try:
        analytics = await db_manager.get_feedback_analytics(days)
        return FeedbackAnalytics(**analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback analytics: {e}")

# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/analytics/system", response_model=SystemAnalytics)
async def get_system_analytics(current_user: dict = Depends(get_current_user)):
    """Get comprehensive system analytics"""
    try:
        analytics = await db_manager.get_system_analytics()
        return SystemAnalytics(**analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system analytics: {e}")

@app.get("/analytics/agents")
async def get_agent_analytics(current_user: dict = Depends(get_current_user)):
    """Get multi-agent system analytics"""
    try:
        agent_status = orchestration_agent.get_agent_status()
        vector_analytics = await vector_store.get_vector_analytics()
        
        return {
            "agent_system": agent_status,
            "vector_store": vector_analytics,
            "processing_stats": {
                "total_processed_documents": await db_manager.get_document_count(),
                "total_vector_chunks": vector_analytics["total_chunks"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent analytics: {e}")

# ============================================================================
# AGENT-SPECIFIC ENDPOINTS
# ============================================================================

@app.post("/agents/analyze", response_model=AgentResponse)
async def agent_analysis(
    request: AgentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Direct agent analysis endpoint"""
    try:
        response = await orchestration_agent.process(request.dict())
        return AgentResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {e}")

@app.get("/agents/status", response_model=AgentStatusResponse)
async def get_agents_status():
    """Get status of all agents"""
    try:
        status = orchestration_agent.get_agent_status()
        return AgentStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {e}")

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/sectors")
async def get_sectors():
    """Get available sectors"""
    try:
        sectors = await db_manager.get_sectors()
        return {"sectors": sectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sectors: {e}")

@app.get("/use-cases")
async def get_use_cases(sector: Optional[str] = None):
    """Get available use cases"""
    try:
        use_cases = await db_manager.get_use_cases(sector)
        return {"use_cases": use_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get use cases: {e}")

if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT) 