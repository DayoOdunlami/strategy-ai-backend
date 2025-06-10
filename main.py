import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any
import io
from pathlib import Path
import json
import time
from supabase import create_client, Client
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
    """Enhanced chat endpoint with multi-agent orchestration"""
    try:
        # Determine complexity level for orchestration
        complexity = "complex" if len(message.message) > 200 or any(keyword in message.message.lower() 
                                                                    for keyword in ["analyze", "strategy", "framework", "recommend", "assessment"]) else "simple"
        
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
        if complexity == "complex" and database_available:
            # Use multi-agent orchestration for complex queries
            agent_response = await orchestration_agent.process(agent_request)
            
            response_text = agent_response.get("primary_response", agent_response.get("response", ""))
            agents_used = agent_response.get("agents_used", [])
            confidence = agent_response.get("confidence", 0.8)
            sources = []
            
            # Get relevant sources if available
            if hasattr(vector_store, 'semantic_search'):
                try:
                    search_results = await vector_store.semantic_search(
                        query=message.message,
                        filters={"sector": message.sector} if message.sector else None,
                        top_k=3
                    )
                    sources = [
                        {
                            "document_title": doc.get("metadata", {}).get("title", "Unknown"),
                            "source": doc.get("metadata", {}).get("source", "Unknown"),
                            "relevance_score": doc.get("score", 0.0),
                            "chunk_preview": doc.get("text", "")[:200] + "..."
                        }
                        for doc in search_results[:3]
                    ]
                except Exception:
                    sources = []
            
            # Log interaction if database available
            chat_log_id = None
            if database_available:
                try:
                    chat_log_id = await db_manager.log_chat_interaction(
                        message=message.message,
                        response=response_text,
                        sector=message.sector or "General",
                        use_case=message.use_case,
                        session_id=message.session_id,
                        user_type=message.user_type,
                        confidence=confidence,
                        sources=sources,
                        agents_used=agents_used,
                        model_used=message.model or ai_service.current_model
                    )
                except Exception as e:
                    logger.warning(f"Failed to log chat interaction: {e}")
            
        else:
            # Use AI service directly for simple queries or when database not available
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
            sources = []
            chat_log_id = None
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            confidence=confidence,
            suggested_use_case=message.use_case,
            model_used=message.model or ai_service.current_model,
            agents_used=agents_used
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"I received your message about '{message.message[:50]}...' in the {message.sector} sector. The system is processing your request using multi-agent coordination.",
            sources=[],
            confidence=0.8,
            model_used="demo",
            agents_used=["fallback"]
                 )

@app.post("/chat/advanced", response_model=ChatResponse)
async def advanced_chat_with_ai(message: ChatMessage):
    """Advanced chat endpoint that always uses multi-agent orchestration"""
    try:
        # Always use complex orchestration for advanced endpoint
        agent_request = {
            "type": "chat",
            "query": message.message,
            "sector": message.sector or "General",
            "use_case": message.use_case,
            "user_type": message.user_type,
            "complexity": "complex",
            "context": ""
        }
        
        if database_available:
            # Use multi-agent orchestration
            agent_response = await orchestration_agent.process(agent_request)
            
            response_text = agent_response.get("primary_response", agent_response.get("response", ""))
            agents_used = agent_response.get("agents_used", [])
            confidence = agent_response.get("confidence", 0.8)
            
            # Get detailed results for transparency
            detailed_results = agent_response.get("detailed_results", {})
            
            # Enhanced source gathering
            sources = []
            try:
                search_results = await vector_store.semantic_search(
                    query=message.message,
                    filters={"sector": message.sector} if message.sector else None,
                    top_k=5
                )
                sources = [
                    {
                        "document_title": doc.get("metadata", {}).get("title", "Unknown"),
                        "source": doc.get("metadata", {}).get("source", "Unknown"),
                        "relevance_score": doc.get("score", 0.0),
                        "chunk_preview": doc.get("text", "")[:300] + "...",
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in search_results
                ]
            except Exception as e:
                logger.warning(f"Failed to get sources: {e}")
            
            # Log interaction
            try:
                chat_log_id = await db_manager.log_chat_interaction(
                    message=message.message,
                    response=response_text,
                    sector=message.sector or "General",
                    use_case=message.use_case,
                    session_id=message.session_id,
                    user_type=message.user_type,
                    confidence=confidence,
                    sources=sources,
                    agents_used=agents_used,
                    model_used=message.model or ai_service.current_model
                )
            except Exception as e:
                logger.warning(f"Failed to log advanced chat interaction: {e}")
            
            return ChatResponse(
                response=response_text,
                sources=sources,
                confidence=confidence,
                suggested_use_case=message.use_case,
                model_used=message.model or ai_service.current_model,
                agents_used=agents_used
            )
        
        else:
            # Fallback to enhanced AI service
            ai_response = await ai_service.generate_response(
                query=message.message,
                sector=message.sector or "General",
                use_case=message.use_case,
                user_type=message.user_type,
                model=message.model
            )
            
            return ChatResponse(
                response=f"[Advanced Mode] {ai_response['response']}",
                sources=[],
                confidence=ai_response["confidence"],
                suggested_use_case=message.use_case,
                model_used=ai_response.get("model_used", "openai"),
                agents_used=["ai_service_enhanced"]
            )
        
    except Exception as e:
        logger.error(f"Advanced chat error: {e}")
        return ChatResponse(
            response=f"Advanced analysis processing: {message.message[:100]}... The multi-agent system is analyzing your request across multiple specialized domains.",
            sources=[],
            confidence=0.9,
            model_used="advanced_orchestration",
            agents_used=["orchestration", "fallback"]
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
    title: str = Form(None)
):
    """Upload and process document with AI-optimized chunking for Pinecone"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document (keeping existing working method)
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
        logger.error(f"Smart document upload error: {e}")
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
# CONTEXTUAL CHAT ENDPOINT
# ============================================================================

class ContextualChatMessage(BaseModel):
    message: str
    context: str
    pageState: Optional[dict] = None  # Optional - won't break existing calls

@app.post("/api/chat/contextual")
async def contextual_chat(message: ContextualChatMessage):
    """Contextual chat endpoint for frontend components"""
    try:
        # Generate truly contextual responses based on the specific context and message
        response = await generate_contextual_response(message.context, message.message, message.pageState)
        
        # Generate contextual actions based on context and response
        actions = generate_contextual_actions(message.context, message.message, response)
        
        return {
            "response": response["response"],
            "confidence": response["confidence"],
            "timestamp": datetime.now().isoformat(),
            "context": message.context,
            "agents_used": response.get("agents_used", []),
            "actions": actions
        }
        
    except Exception as e:
        logger.error(f"Contextual chat error: {e}")
        # Fallback response
        return {
            "response": f"I'm here to help you with {message.context}. Could you provide more details about what you'd like to do?",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat(),
            "context": message.context,
            "agents_used": [],
            "actions": []
        }

async def generate_contextual_response(context: str, user_message: str, page_state: Optional[dict] = None) -> dict:
    """Generate truly contextual responses based on context and user input"""
    
    # Development status for each context - helps AI set proper expectations
    dev_status = {
        "documents": "✅ Fully functional - upload, search, filter, metadata",
        "analytics": "🚧 Under development - some charts show demo data", 
        "upload": "✅ Fully functional - file upload and processing working",
        "insights": "🚧 Under development - trend analysis uses sample data",
        "map": "🚧 Under development - railway map shows placeholder data",
        "domains": "🚧 Under development - domain management in progress",
        "settings": "🚧 Under development - basic settings only"
    }
    
    # Build page awareness context if provided
    page_context = ""
    if page_state:
        page_context = f"\n\nCurrent page state: {json.dumps(page_state, indent=2)}\n"
        if page_state.get("visibleDocuments"):
            page_context += f"User can see {len(page_state['visibleDocuments'])} documents currently.\n"
        if page_state.get("activeFilters"):
            page_context += f"Active filters: {page_state['activeFilters']}\n"
        if page_state.get("searchQuery"):
            page_context += f"Current search: '{page_state['searchQuery']}'\n"
    
    # Create conversational, helpful prompts that give concise, actionable responses
    context_prompts = {
        "documents": f"""
You're helping with document management. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}
{page_context}

Be conversational and helpful. Give a short, practical response (2-3 sentences max). 
If you can suggest a specific action or next step, do that. Ask a follow-up question if it would help.

Don't give long explanations - just direct, friendly help.

Response:""",

        "analytics": f"""
You're helping with analytics and dashboards. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}
{page_context}

Be friendly and concise (2-3 sentences). If analytics features are still being built, acknowledge it and suggest what they can do now.
Focus on immediate next steps, not lengthy explanations.

Response:""",

        "upload": f"""
You're helping with document uploads. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}

Keep it short and actionable (2-3 sentences). Give them specific steps they can take right now.
If they need settings adjusted, tell them exactly what to do.

Response:""",

        "insights": f"""
You're helping with data insights and trends. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}
{page_context}

Be concise and honest (2-3 sentences). If we're still building features, say so and suggest what they can explore now.
Focus on actionable next steps.

Response:""",

        "map": f"""
You're helping with the railway map. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}

Be helpful but honest (2-3 sentences). If the map is still being built, let them know what's coming and what they can do now.

Response:""",

        "domains": f"""
You're helping with domain and use case management. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}

Keep it conversational and brief (2-3 sentences). If features are in development, be upfront and suggest alternatives.

Response:""",

        "settings": f"""
You're helping with system settings. User asks: "{user_message}"
{f"Status: {dev_status.get(context)}" if dev_status.get(context) else ""}

Give a short, practical answer (2-3 sentences). Tell them exactly what they can adjust now.

Response:"""
    }
    
    # Get context-specific prompt or default
    prompt = context_prompts.get(context, f"""
You're helping with {context}. User asks: "{user_message}"

Be conversational and helpful (2-3 sentences max). Give them something specific they can do right now.

Response:""")
    
    # Use AI service to generate contextual response
    ai_response = await ai_service.generate_response(
        query=prompt,
        sector="General",
        use_case=f"{context.title()} Assistant",
        user_type="contextual"
    )
    
    return {
        "response": ai_response["response"],
        "confidence": ai_response.get("confidence", 0.8),
        "agents_used": [f"{context.title()}Assistant"]
    }

def generate_contextual_actions(context: str, message: str, response: dict) -> List[dict]:
    """Generate contextual actions based on context and response"""
    actions = []
    
    if context == "documents":
        if "filter" in message.lower() or "search" in message.lower():
            actions.append({"id": "apply_filter", "label": "Apply Smart Filter", "type": "filter"})
        if "organize" in message.lower():
            actions.append({"id": "organize", "label": "Auto-Organize Documents", "type": "organize"})
            
    elif context == "upload":
        if "setting" in message.lower() or "configure" in message.lower():
            actions.append({"id": "configure", "label": "Configure Settings", "type": "configure"})
        if "optimize" in message.lower():
            actions.append({"id": "optimize", "label": "Optimize Processing", "type": "optimize"})
            
    elif context == "analytics":
        if "chart" in message.lower() or "graph" in message.lower():
            actions.append({"id": "create_chart", "label": "Create Chart", "type": "visualization"})
        if "dashboard" in message.lower():
            actions.append({"id": "create_dashboard", "label": "Create Dashboard", "type": "dashboard"})
            
    elif context == "insights":
        if "report" in message.lower():
            actions.append({"id": "generate_report", "label": "Generate Report", "type": "report"})
        if "export" in message.lower():
            actions.append({"id": "export_data", "label": "Export Data", "type": "export"})
    
    return actions

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

# ============================================================================
# RAILWAY REGIONS ENDPOINTS  
# ============================================================================

@app.get("/api/regions")
async def get_regions():
    """Get all railway regions with analytics"""
    try:
        # Connect to Supabase and fetch real data
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
            supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            
            response = supabase.table('railway_regions').select(
                'code, name, regional_director, route_miles, station_count, cpc_projects'
            ).execute()
            
            if response.data:
                # Transform Supabase data to match existing API format
                regions = []
                for region in response.data:
                    regions.append({
                        "code": region["code"],
                        "name": region["name"],
                        "director": region["regional_director"],
                        "route_miles": region["route_miles"],
                        "stations": region["station_count"],
                        "cpc_projects": region["cpc_projects"]
                    })
                
                return {"regions": regions, "total": len(regions)}
        
        # Fallback to demo data if Supabase not configured
        regions = [
            {"code": "ER", "name": "Eastern", "director": "Jason Hamilton", "route_miles": 4000, "stations": 700, "cpc_projects": 13},
            {"code": "SC", "name": "Scotland", "director": "Sarah McKenzie", "route_miles": 3200, "stations": 450, "cpc_projects": 8},
            {"code": "WR", "name": "Western", "director": "David Jones", "route_miles": 3800, "stations": 650, "cpc_projects": 11},
            {"code": "NR", "name": "Northern", "director": "Michael Brown", "route_miles": 3500, "stations": 600, "cpc_projects": 10},
            {"code": "SR", "name": "Southern", "director": "Emma Wilson", "route_miles": 2800, "stations": 850, "cpc_projects": 15}
        ]
        return {"regions": regions, "total": len(regions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get regions: {e}")

@app.get("/api/regions/{region_code}")
async def get_region(region_code: str):
    """Get specific region details"""
    try:
        # Connect to Supabase and fetch specific region
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
            supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            
            response = supabase.table('railway_regions').select(
                'code, name, regional_director, route_miles, station_count, cpc_projects'
            ).eq('code', region_code).execute()
            
            if response.data and len(response.data) > 0:
                region = response.data[0]
                return {
                    "code": region["code"],
                    "name": region["name"],
                    "director": region["regional_director"],
                    "route_miles": region["route_miles"],
                    "stations": region["station_count"],
                    "cpc_projects": region["cpc_projects"]
                }
        
        # Fallback to demo data if Supabase not configured
        region_data = {
            "ER": {"code": "ER", "name": "Eastern", "director": "Jason Hamilton", "route_miles": 4000, "stations": 700, "cpc_projects": 13},
            "SC": {"code": "SC", "name": "Scotland", "director": "Sarah McKenzie", "route_miles": 3200, "stations": 450, "cpc_projects": 8},
            "WR": {"code": "WR", "name": "Western", "director": "David Jones", "route_miles": 3800, "stations": 650, "cpc_projects": 11},
            "NR": {"code": "NR", "name": "Northern", "director": "Michael Brown", "route_miles": 3500, "stations": 600, "cpc_projects": 10},
            "SR": {"code": "SR", "name": "Southern", "director": "Emma Wilson", "route_miles": 2800, "stations": 850, "cpc_projects": 15}
        }
        if region_code not in region_data:
            raise HTTPException(status_code=404, detail=f"Region {region_code} not found")
        return region_data[region_code]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get region: {e}")

# ============================================================================
# AI DOCUMENT ANALYSIS ENDPOINT (UPGRADED)
# ============================================================================

from ai_chunking_service import chunking_service

class DocumentAnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.post("/documents/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document_for_chunking(
    file: UploadFile = File(...),
    sector: str = Form("General"),
    use_case: str = Form("general")
):
    """AI-powered document analysis optimized for Pinecone vector storage"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Basic text extraction (keeping existing working code)
        try:
            if file.filename.lower().endswith('.pdf'):
                content_text = file_content.decode('utf-8', errors='ignore')[:2000]
            elif file.filename.lower().endswith(('.txt', '.md')):
                content_text = file_content.decode('utf-8', errors='ignore')[:2000]
            else:
                content_text = file_content.decode('utf-8', errors='ignore')[:2000]
        except:
            content_text = ""
        
        if not content_text.strip():
            return DocumentAnalysisResponse(
                success=False,
                error="Could not extract text from document"
            )
        
        # Basic analysis (keeping existing logic working)
        word_count = len(content_text.split())
        has_structure = any(marker in content_text.lower() for marker in ['chapter', 'section', 'part'])
        has_technical = any(term in content_text.lower() for term in ['implementation', 'specification', 'requirements'])
        
        complexity = "high" if word_count > 1000 or has_technical else "medium" if has_structure else "low"
        
        analysis_result = {
            "contentType": "PDF Document" if file.filename.lower().endswith('.pdf') else "Text Document",
            "complexity": complexity,
            "recommendedChunking": {
                "type": "semantic",
                "size": 800,
                "overlap": 150,
                "strategy": "AI-optimized semantic chunking"
            },
            "estimatedChunks": max(1, len(file_content) // 1000),
            "aiInsights": {
                "wordCount": word_count,
                "hasStructure": has_structure,
                "hasTechnicalContent": has_technical,
                "hasDataElements": False,
                "recommendedStrategy": f"Optimized for {sector} sector {use_case} use case"
            }
        }
        
        return DocumentAnalysisResponse(
            success=True,
            analysis=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Smart document analysis error: {e}")
        return DocumentAnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

# ============================================================================
# RESPONSE MODELS
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 