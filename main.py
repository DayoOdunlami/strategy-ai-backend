import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging
from datetime import datetime
import uuid
from typing import List, Optional

# Import our custom modules
from config import Settings
from auth import get_current_user, create_access_token, verify_password, get_password_hash
from models import (
    ChatMessage, ChatResponse, DocumentUpload, DocumentResponse, 
    DocumentListResponse, SearchFilter, SearchResult, SearchResponse,
    SystemAnalytics, UserFeedback, FeedbackResponse, ProcessingStatus,
    APIError, SystemSettings
)

# Initialize settings
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server Configuration
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))  # Properly handle Railway's PORT

app = FastAPI(
    title="Strategy AI Backend",
    description="Enhanced FastAPI backend for Strategy AI platform with document processing, chat, and analytics",
    version="2.1.1"
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
# BASIC ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Strategy AI Backend is running!",
        "version": "2.1.1",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "enhancement": "Authentication system enabled",
        "features": ["authentication", "chat", "document_management", "search", "analytics", "feedback"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "local"),
        "version": "2.1.1"
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

@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """Example of a protected route"""
    return {
        "message": f"Hello {current_user['sub']}, this is a protected route!",
        "user": current_user,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """AI Chat endpoint (mock implementation)"""
    # Mock AI response for demonstration
    chat_id = str(uuid.uuid4())
    
    response_text = f"Hello! I received your message about '{message.message[:50]}...' in the {message.sector} sector. This is a mock response. In production, this would connect to AI services."
    
    # Store chat log
    chat_logs[chat_id] = {
        "id": chat_id,
        "message": message.message,
        "response": response_text,
        "sector": message.sector,
        "timestamp": datetime.now(),
        "user_type": message.user_type
    }
    
    return ChatResponse(
        response=response_text,
        sources=[
            {"title": "Mock Document 1", "relevance": 0.9},
            {"title": "Mock Document 2", "relevance": 0.8}
        ],
        confidence=0.85,
        suggested_use_case=f"{message.sector} Analysis" if message.sector != "General" else None
    )

@app.get("/chat/history")
async def get_chat_history(limit: int = 10, current_user: dict = Depends(get_current_user)):
    """Get recent chat history"""
    recent_chats = list(chat_logs.values())[-limit:]
    return {
        "chats": recent_chats,
        "total_count": len(chat_logs),
        "limit": limit
    }

# ============================================================================
# DOCUMENT ENDPOINTS
# ============================================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    sector: str = "General",
    use_case: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Upload a document (mock implementation)"""
    doc_id = str(uuid.uuid4())
    
    # Mock document processing
    document = {
        "id": doc_id,
        "title": title or file.filename,
        "filename": file.filename,
        "sector": sector,
        "use_case": use_case,
        "tags": None,
        "source_type": "file",
        "status": "completed",
        "chunk_count": 5,  # Mock chunk count
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    documents_store[doc_id] = document
    
    return DocumentResponse(**document)

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 10, 
    offset: int = 0,
    sector: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """List documents with optional filtering"""
    all_docs = list(documents_store.values())
    
    # Filter by sector if specified
    if sector:
        all_docs = [doc for doc in all_docs if doc["sector"] == sector]
    
    total_count = len(all_docs)
    docs = all_docs[offset:offset + limit]
    
    return DocumentListResponse(
        documents=[DocumentResponse(**doc) for doc in docs],
        total_count=total_count,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total_count
    )

@app.get("/documents/{document_id}")
async def get_document(document_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific document"""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(**documents_store[document_id])

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a document"""
    if document_id not in documents_store:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del documents_store[document_id]
    return {"message": f"Document {document_id} deleted successfully"}

# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search_documents(search_filter: SearchFilter):
    """Search through documents (mock implementation)"""
    # Mock search results
    mock_results = [
        SearchResult(
            document_id="doc_1",
            title="Strategy Document 1",
            chunk_text="This is a sample chunk of text that matches your search criteria...",
            relevance_score=0.95,
            metadata={"sector": search_filter.sector or "General", "use_case": "analysis"},
            highlighted_text="sample <mark>chunk</mark> of text"
        ),
        SearchResult(
            document_id="doc_2", 
            title="Policy Analysis Report",
            chunk_text="Another relevant piece of content from our document collection...",
            relevance_score=0.87,
            metadata={"sector": search_filter.sector or "Transport", "use_case": "policy"},
            highlighted_text="relevant piece of <mark>content</mark>"
        )
    ]
    
    return SearchResponse(
        results=mock_results,
        total_count=len(mock_results),
        search_time_ms=45.5,
        filters_applied=search_filter,
        suggestions=["Try searching for 'policy analysis'", "Consider using 'transport' sector"]
    )

# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: UserFeedback):
    """Submit user feedback"""
    feedback_id = str(uuid.uuid4())
    
    feedback_entry = {
        "id": feedback_id,
        "chat_log_id": feedback.chat_log_id,
        "document_id": feedback.document_id,
        "session_id": feedback.session_id,
        "rating": feedback.rating,
        "feedback_type": feedback.feedback_type,
        "comment": feedback.comment,
        "helpful": feedback.helpful,
        "timestamp": datetime.now()
    }
    
    feedback_store[feedback_id] = feedback_entry
    
    return FeedbackResponse(
        success=True,
        feedback_id=feedback_id,
        message="Thank you for your feedback!"
    )

@app.get("/feedback/analytics")
async def get_feedback_analytics(current_user: dict = Depends(get_current_user)):
    """Get feedback analytics (admin only)"""
    if not feedback_store:
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "helpful_percentage": 0.0,
            "rating_distribution": {}
        }
    
    ratings = [f["rating"] for f in feedback_store.values() if f["rating"]]
    helpful_count = sum(1 for f in feedback_store.values() if f["helpful"] is True)
    
    return {
        "total_feedback": len(feedback_store),
        "average_rating": sum(ratings) / len(ratings) if ratings else 0.0,
        "helpful_percentage": (helpful_count / len(feedback_store)) * 100 if feedback_store else 0.0,
        "rating_distribution": {str(i): ratings.count(i) for i in range(1, 6)}
    }

# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/analytics/system", response_model=SystemAnalytics)
async def get_system_analytics(current_user: dict = Depends(get_current_user)):
    """Get system analytics"""
    sectors = {}
    use_cases = {}
    
    for doc in documents_store.values():
        sector = doc["sector"]
        use_case = doc["use_case"]
        
        sectors[sector] = sectors.get(sector, 0) + 1
        if use_case:
            use_cases[use_case] = use_cases.get(use_case, 0) + 1
    
    return SystemAnalytics(
        total_documents=len(documents_store),
        total_chunks=sum(doc["chunk_count"] for doc in documents_store.values()),
        total_sectors=len(sectors),
        total_use_cases=len(use_cases),
        documents_by_sector=sectors,
        documents_by_use_case=use_cases,
        recent_activity_count=len(chat_logs),
        storage_usage={"documents": len(documents_store), "feedback": len(feedback_store)}
    )

@app.get("/settings", response_model=SystemSettings)
async def get_system_settings(current_user: dict = Depends(get_current_user)):
    """Get system settings"""
    return SystemSettings()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT) 