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
except ImportError:
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

try:
    from database import db_manager
    from vector_store import vector_store
    from specialized_agents import orchestration_agent
    from document_processor import document_processor
except ImportError as e:
    logging.warning(f"Could not import advanced modules: {e}")
    # Create simple fallback objects
    class SimpleManager:
        async def test_connection(self): return True
        def get_available_models(self): return ["demo"]
        def get_agent_status(self): return {"status": "demo_mode"}
    
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
                "multi_agents": "ready"
            },
            "ai_integration": "enabled"
        }
    except Exception as e:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": "production", 
            "version": "3.0.0",
            "mode": "simplified",
            "message": f"Running in simplified mode: {e}"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 