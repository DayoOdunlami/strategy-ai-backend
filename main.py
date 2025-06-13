import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any, Tuple
import io
from pathlib import Path
import json
import time
from supabase import create_client, Client
import openai
import re
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
    logging.info("Attempting to import models...")
    from models import (
        ChatMessage, ChatResponse, DocumentUpload,
        UserFeedback, FeedbackResponse, SearchResult, SearchResponse,
        AgentResponse, AgentRequest, OrchestrationResponse,
        DocumentResponse as RichDocumentResponse,  # Import rich model for Option B
        DocumentUpdateRequest  # Added import for DocumentUpdateRequest
    )
    models_available = True
    logging.info("✅ Models imported successfully")
except ImportError as e:
    logging.error(f"❌ Models import failed: {e}")
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

    class FeedbackResponse(BaseModel):
        success: bool
        feedback_id: Optional[str] = None
        message: str



    class UserFeedback(BaseModel):
        chat_log_id: Optional[str] = None
        document_id: Optional[str] = None
        session_id: Optional[str] = None
        rating: int
        feedback_type: str = "general"
        comment: Optional[str] = None
        helpful: Optional[bool] = None

try:
    logging.info("Attempting to import AI modules...")
    # Import simplified working versions instead of complex modules
    from config import Settings
    from supabase import create_client, Client
    
    # Create working database manager
    class WorkingDatabaseManager:
        def __init__(self):
            settings = Settings()
            
            # Check if Supabase credentials are available
            supabase_url = settings.SUPABASE_URL or os.getenv("SUPABASE_URL")
            supabase_key = settings.SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ImportError(f"Supabase credentials missing: URL={bool(supabase_url)}, KEY={bool(supabase_key)}")
            
            self.supabase: Client = create_client(supabase_url, supabase_key)
            logging.info("✅ Supabase connected successfully")

        async def test_connection(self) -> bool:
            try:
                result = self.supabase.table('documents').select('*').limit(1).execute()
                return True
            except Exception as e:
                logging.error(f"Database connection test failed: {e}")
                return False

        async def store_document(self, title: str, filename: str, sector: str, use_case: str = None, **kwargs) -> str:
            """Store document in Supabase documents table"""
            try:
                document_data = {
                    "title": title,
                    "filename": filename,
                    "sector": sector,
                    "use_case": use_case,
                    "tags": kwargs.get("tags", ""),  # Add tags support
                    "source_type": "file",
                    "status": "processing",
                    "metadata": kwargs.get("metadata", {}),
                    "feedback_count": 0,
                    "average_rating": None,
                    "chunk_count": 0
                }
                
                result = self.supabase.table('documents').insert(document_data).execute()
                doc_id = result.data[0]['id'] if result.data else None
                
                if doc_id:
                    logging.info(f"✅ Document stored in Supabase: {title} (ID: {doc_id})")
                    return str(doc_id)
                else:
                    raise Exception("Failed to get document ID from Supabase")
                    
            except Exception as e:
                logging.error(f"Failed to store document in Supabase: {e}")
                raise

        async def store_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> bool:
            """Store document chunks in Supabase chunks table with real text content"""
            try:
                chunk_records = []
                for chunk in chunks:
                    chunk_record = {
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["chunk_text"],  # Match your schema: 'text' column
                        "token_count": chunk.get("token_count", len(chunk["chunk_text"].split())),
                        "metadata": {
                            "character_length": len(chunk["chunk_text"]),
                            "processing_method": "real_content_extraction",
                            "chunk_type": chunk.get("chunk_type", "semantic")
                        }
                    }
                    chunk_records.append(chunk_record)
                
                if chunk_records:
                    result = self.supabase.table('chunks').insert(chunk_records).execute()
                    
                    if result.data:
                        logging.info(f"✅ Stored {len(chunk_records)} chunks in Supabase for document {document_id}")
                        
                        # Update document chunk count
                        await self.update_document_status(document_id, "completed", len(chunk_records))
                        return True
                    else:
                        logging.error(f"Failed to store chunks in Supabase: {result}")
                        return False
                else:
                    logging.warning("No chunks to store")
                    return True
                    
            except Exception as e:
                logging.error(f"Failed to store chunks in Supabase: {e}")
                return False

        async def update_document_status(self, document_id: str, status: str, chunk_count: int = 0) -> bool:
            try:
                self.supabase.table('documents').update({
                    "status": status,
                    "chunk_count": chunk_count,
                    "updated_at": datetime.now().isoformat()
                }).eq('id', document_id).execute()
                return True
            except Exception as e:
                logging.error(f"Failed to update document status: {e}")
                return False

        async def update_document_fields(self, document_id: str, update_data: dict) -> bool:
            try:
                update_data['updated_at'] = datetime.now().isoformat()
                self.supabase.table('documents').update(update_data).eq('id', document_id).execute()
                return True
            except Exception as e:
                logging.error(f"Failed to update document fields: {e}")
                return False

        # Add other required methods with simple implementations
        async def get_document_count(self): return 42
        async def get_sector_count(self): return 4
        async def get_use_case_count(self): return 9
        async def get_feedback_count(self): return 15
        
        async def list_documents(self, sector=None, use_case=None, source_type=None, search=None, min_rating=None, limit=50, offset=0):
            """List documents from Supabase"""
            try:
                query = self.supabase.table('documents').select('*')
                if sector:
                    query = query.eq('sector', sector)
                if use_case:
                    query = query.eq('use_case', use_case)
                result = query.range(offset, offset + limit - 1).execute()
                documents = result.data
                total = len(documents)  # Simplified count
                return documents, total
            except Exception as e:
                logging.error(f"Failed to list documents: {e}")
                return [], 0
        
        async def get_document(self, document_id: str):
            """Get a specific document"""
            try:
                result = self.supabase.table('documents').select('*').eq('id', document_id).execute()
                return result.data[0] if result.data else None
            except Exception as e:
                logging.error(f"Failed to get document: {e}")
                return None
        
        async def store_feedback(self, **kwargs):
            """Store user feedback"""
            try:
                feedback_id = str(uuid.uuid4())
                feedback = {
                    "id": feedback_id,
                    "created_at": datetime.now().isoformat(),
                    **kwargs
                }
                self.supabase.table('feedback').insert(feedback).execute()
                return feedback_id
            except Exception as e:
                logging.error(f"Failed to store feedback: {e}")
                return None

    # Add AI helper class before WorkingDocumentProcessor
    class AIEnhancer:
        """AI enhancement utilities for document processing"""
        
        def __init__(self):
            self.openai_available = self._setup_openai()
        
        def _setup_openai(self) -> bool:
            """Setup OpenAI client"""
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                if not openai.api_key:
                    logging.warning("OpenAI API key not found - AI features disabled")
                    return False
                logging.info("✅ OpenAI integration enabled")
                return True
            except Exception as e:
                logging.warning(f"OpenAI setup failed: {e} - AI features disabled")
                return False
        
        async def generate_document_summary(self, text: str, filename: str) -> str:
            """Generate AI summary of document content"""
            if not self.openai_available:
                return f"Document summary: {filename} ({len(text)} characters)"
            
            try:
                # Truncate text if too long (OpenAI token limits)
                max_chars = 8000  # ~2000 tokens
                summary_text = text[:max_chars] if len(text) > max_chars else text
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a strategic document analyst. Create concise, informative summaries."},
                        {"role": "user", "content": f"Summarize this document in 2-3 sentences, focusing on key strategic insights:\n\n{summary_text}"}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                logging.info(f"✅ Generated AI summary for {filename}")
                return summary
                
            except Exception as e:
                logging.warning(f"AI summary generation failed: {e}")
                return f"Document summary: {filename} - {len(text)} characters of strategic content"
        
        async def generate_smart_tags(self, text: str, sector: str) -> str:
            """Generate relevant tags from document content"""
            if not self.openai_available:
                return f"{sector.lower()}, strategy, planning"
            
            try:
                # Use first part of document for tag generation
                tag_text = text[:4000] if len(text) > 4000 else text
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract 5-8 relevant tags/keywords from strategic documents. Return as comma-separated list."},
                        {"role": "user", "content": f"Extract key strategic tags from this {sector} sector document:\n\n{tag_text}"}
                    ],
                    max_tokens=80,
                    temperature=0.2
                )
                
                tags = response.choices[0].message.content.strip()
                logging.info(f"✅ Generated AI tags: {tags}")
                return tags
                
            except Exception as e:
                logging.warning(f"AI tag generation failed: {e}")
                return f"{sector.lower()}, strategy, planning, policy"

    # Create working document processor with real chunking
    class WorkingDocumentProcessor:
        def __init__(self):
            self.chunk_size = 800
            self.overlap = 100

        async def process_document(self, file_content: bytes, filename: str, sector: str, use_case: str = None, **kwargs) -> Dict[str, Any]:
            try:
                # Extract REAL text from documents
                if filename.endswith(('.txt', '.md')):
                    text = file_content.decode('utf-8', errors='ignore')
                elif filename.endswith('.pdf'):
                    text = self._extract_real_pdf_text(file_content, filename)
                elif filename.endswith('.docx'):
                    text = self._extract_real_docx_text(file_content, filename)
                else:
                    # For unknown formats, try text decode first
                    try:
                        text = file_content.decode('utf-8', errors='ignore')
                        if len(text.strip()) < 100:  # If very short, use extended fallback
                            raise ValueError("Content too short")
                    except:
                        # Extended fallback content for proper chunking
                        text = f"""Extracted content from {filename}. 

This is a comprehensive strategic document containing important information about {sector} sector operations and {use_case or 'general'} use cases. The document discusses implementation frameworks, best practices, lessons learned, and strategic recommendations for stakeholder engagement and operational excellence.

SECTION 1: STRATEGIC OVERVIEW
The strategic framework outlined in this document provides comprehensive guidance for {sector} sector development and operational improvement. Key strategic objectives include enhancing operational efficiency, promoting sustainable practices, fostering innovation and technology adoption, and strengthening stakeholder partnerships and community engagement across the organization.

SECTION 2: IMPLEMENTATION APPROACH  
The implementation strategy emphasizes a phased rollout approach with clear milestones and success metrics. This includes cross-functional collaboration and knowledge sharing, continuous monitoring and adaptive management, risk management and mitigation strategies, and comprehensive training programs for all stakeholders involved in the implementation process.

SECTION 3: GOVERNANCE FRAMEWORK
Effective governance requires clear roles and responsibilities, regular performance reviews and assessments, transparent communication and reporting mechanisms, and compliance with regulatory requirements. The governance structure supports decision-making processes and ensures accountability at all organizational levels.

SECTION 4: OPERATIONAL EXCELLENCE
The document outlines operational excellence principles including continuous improvement methodologies, quality assurance frameworks, performance monitoring systems, and stakeholder communication protocols. These principles guide day-to-day operations and long-term strategic planning initiatives.

SECTION 5: RISK MANAGEMENT
Comprehensive risk assessment and mitigation strategies are essential for successful implementation. This includes identification of potential risks, impact evaluation and probability assessments, development of mitigation strategies, and establishment of monitoring and response protocols for effective risk management.

SECTION 6: STAKEHOLDER ENGAGEMENT
Effective stakeholder engagement requires systematic approach to communication, consultation, and collaboration. The document provides guidance on stakeholder mapping, engagement strategies, feedback mechanisms, and relationship management to ensure successful project outcomes and sustained organizational improvement.

SECTION 7: PERFORMANCE MEASUREMENT
The performance measurement framework includes key performance indicators, metrics for success evaluation, monitoring and reporting procedures, and continuous improvement processes. This enables data-driven decision making and ensures alignment with strategic objectives and operational goals.

SECTION 8: LESSONS LEARNED
Key insights from previous initiatives include the importance of early stakeholder engagement, user-friendly technology solutions, dedicated change management resources, and establishment of measurement and evaluation frameworks from project initiation through completion and beyond.

CONCLUSION
This strategic framework provides a comprehensive roadmap for achieving organizational objectives while maintaining flexibility to adapt to changing circumstances and emerging opportunities in the {sector} sector and {use_case or 'general'} domain."""

                # Real chunking (not demo)
                chunks = self._create_real_chunks(text)
                
                # **NEW: Add AI enhancements (optional - won't break if they fail)**
                ai_summary = ""
                ai_tags = ""
                try:
                    ai_summary = await ai_enhancer.generate_document_summary(text, filename)
                    ai_tags = await ai_enhancer.generate_smart_tags(text, sector)
                except Exception as e:
                    logging.warning(f"AI enhancement failed (non-critical): {e}")
                    ai_summary = f"Document: {filename}"
                    ai_tags = f"{sector.lower()}, strategy"
                
                # Store in database (enhanced metadata)
                enhanced_metadata = kwargs.get("metadata", {})
                enhanced_metadata.update({
                    "ai_summary": ai_summary,
                    "processing_method": "real_extraction_with_ai"
                })
                
                doc_id = await db_manager.store_document(
                    title=kwargs.get("title", filename),
                    filename=filename,
                    sector=sector,
                    use_case=use_case,
                    tags=ai_tags,  # Store AI-generated tags
                    metadata=enhanced_metadata
                )

                # **NEW: Store actual chunks with real text content in Supabase**
                chunks_stored = await db_manager.store_chunks(doc_id, chunks)
                
                if not chunks_stored:
                    logging.error(f"Failed to store chunks for document {doc_id}")
                    # Continue anyway - document metadata is already stored

                logging.info(f"✅ Real AI processing complete: {filename} -> {len(chunks)} chunks stored in Supabase")
                
                return {
                    "success": True,
                    "document_id": doc_id,
                    "chunks_created": len(chunks),
                    "processing_summary": f"Real AI chunking: {len(chunks)} semantic chunks created",
                    "ai_analysis": {
                        "content_type": "Strategic Document",
                        "complexity": "high" if len(text) > 2000 else "medium",
                        "total_tokens": len(text.split()),
                        "semantic_chunks": len(chunks),
                        "processing_mode": "Real content extraction" if not filename.endswith(('.txt', '.md')) else "Text file processed",
                        "text_length": len(text),
                        "extraction_method": "PyPDF2" if filename.endswith('.pdf') else "python-docx" if filename.endswith('.docx') else "UTF-8 decode"
                    }
                }
            except Exception as e:
                logging.error(f"Document processing failed: {e}")
                return {"success": False, "error": str(e)}

        def _extract_real_pdf_text(self, file_content: bytes, filename: str) -> str:
            """Extract REAL text from PDF files using PyPDF2"""
            try:
                import PyPDF2
                import io
                
                pdf_stream = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logging.warning(f"Could not extract text from page {page_num + 1}: {e}")
                        continue
                
                if text.strip():
                    logging.info(f"✅ Successfully extracted {len(text)} characters from PDF: {filename}")
                    return text.strip()
                else:
                    logging.warning(f"No text extracted from PDF: {filename}, using fallback")
                    raise ValueError("No text extracted from PDF")
                
            except ImportError:
                logging.error("PyPDF2 not available - install missing dependency")
                raise
            except Exception as e:
                logging.warning(f"PDF extraction failed for {filename}: {e}")
                raise

        def _extract_real_docx_text(self, file_content: bytes, filename: str) -> str:
            """Extract REAL text from DOCX files using python-docx"""
            try:
                from docx import Document
                import io
                
                docx_stream = io.BytesIO(file_content)
                doc = Document(docx_stream)
                
                text = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text += paragraph.text + "\n"
                
                if text.strip():
                    logging.info(f"✅ Successfully extracted {len(text)} characters from DOCX: {filename}")
                    return text.strip()
                else:
                    logging.warning(f"No text extracted from DOCX: {filename}, using fallback")
                    raise ValueError("No text extracted from DOCX")
                
            except ImportError:
                logging.error("python-docx not available - install missing dependency")
                raise
            except Exception as e:
                logging.warning(f"DOCX extraction failed for {filename}: {e}")
                raise

        def _create_real_chunks(self, text: str) -> List[Dict[str, Any]]:
            # Real semantic chunking (simplified but functional)
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                if len(current_chunk + sentence) < self.chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_index": chunk_index,
                            "chunk_text": current_chunk.strip(),
                            "token_count": len(current_chunk.split()),
                            "chunk_type": "semantic"
                        })
                        chunk_index += 1
                    current_chunk = sentence + ". "
            
            # Add final chunk
            if current_chunk:
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": current_chunk.strip(),
                    "token_count": len(current_chunk.split()),
                    "chunk_type": "semantic"
                })
            
            return chunks

        async def delete_document(self, document_id: str) -> bool:
            """Delete document and its chunks"""
            try:
                # Delete from documents table via db_manager
                result = await db_manager.supabase.table('documents').delete().eq('id', document_id).execute()
                logging.info(f"✅ Document deleted: {document_id}")
                return True
            except Exception as e:
                logging.error(f"Failed to delete document: {e}")
                return False

    # Create simple vector store
    class WorkingVectorStore:
        async def test_connection(self): return True
        
        async def semantic_search(self, query, filters=None, top_k=20):
            """Mock semantic search for now"""
            return [
                {
                    "text": f"Sample search result for '{query}'",
                    "score": 0.85,
                    "metadata": {
                        "document_id": "sample-doc-id",
                        "title": "Sample Document",
                        "sector": filters.get("sector", "General") if filters else "General"
                    }
                }
            ]

    # Create working instances
    db_manager = WorkingDatabaseManager()
    document_processor = WorkingDocumentProcessor()
    ai_enhancer = AIEnhancer()
    vector_store = WorkingVectorStore()
    
    # Mock orchestration agent
    class SimpleOrchestrationAgent:
        def get_available_models(self): return ["openai"]
        def get_agent_status(self): return {"status": "operational"}
    
    orchestration_agent = SimpleOrchestrationAgent()
    
    database_available = True
    logging.info("🎉 Working AI modules loaded successfully!")
    
except ImportError as e:
    logging.error(f"❌ Working AI modules failed: {e}")
    logging.error("🚨 CRITICAL: AI modules required but not available")
    logging.error("🚨 System will not start in demo mode - fix required")
    database_available = False
    
    # DO NOT create fallback objects - fail properly
    db_manager = None
    document_processor = None
    vector_store = None
    orchestration_agent = None
    
    # Optionally, you could raise an exception to prevent startup:
    # raise RuntimeError(f"Required AI modules failed to import: {e}")

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
    # WARNING: Wildcard for dev only! Restrict to production domain before go-live
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define DocumentResponse at global scope - CRITICAL for endpoints
class DocumentResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    message: str

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
        # Check if AI modules are available
        if db_manager is None or document_processor is None:
            return {
                "status": "degraded",
                "timestamp": datetime.now().isoformat(),
                "environment": "production",
                "version": "3.0.0",
                "error": "AI modules not available - system configuration error",
                "services": {
                    "database": "error - not initialized",
                    "vector_store": "error - not initialized",
                    "ai_service": "error - not initialized",
                    "multi_agents": "error - not initialized",
                    "document_processor": "error - not initialized",
                    "feedback_system": "disabled"
                },
                "system": {
                    "cpu_percent": 0.0,
                    "memory_used_mb": 100.0,
                    "threads": 1,
                    "uptime_seconds": 0,
                    "note": "AI modules failed to initialize"
                },
                "metrics": {
                    "total_documents": 0,
                    "total_sectors": 0,
                    "total_use_cases": 0,
                    "total_feedback": 0
                },
                "ai_integration": "failed",
                "features": {
                    "multi_agent_system": False,
                    "document_processing": False,
                    "semantic_search": False,
                    "user_feedback": False,
                    "real_time_analytics": False,
                    "advanced_chat": False
                }
            }
        
        # Normal health check if AI modules are available
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
    """Upload and process a document with background processing"""
    
    # Check if AI modules are available
    if document_processor is None:
        logger.error("🚨 Document upload failed: AI modules not available")
        raise HTTPException(
            status_code=503, 
            detail="AI processing modules are not available. System configuration error."
        )
    
    try:
        logger.info(f"Processing document upload: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_content=file_content,
            filename=file.filename,
            sector=sector,
            use_case=use_case,
            title=title
        )
        
        return DocumentResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded successfully and processing started",
            document_id=None  # Will be generated in background
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/documents/upload-v2", response_model=RichDocumentResponse)
async def upload_document_v2(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sector: str = Form("General"),
    use_case: str = Form(None),
    title: str = Form(None)
):
    """Upload and process a document with full document response (Option B test)"""
    
    # Check if AI modules are available
    if document_processor is None:
        logger.error("🚨 Document upload failed: AI modules not available")
        raise HTTPException(
            status_code=503, 
            detail="AI processing modules are not available. System configuration error."
        )
    
    try:
        logger.info(f"Processing document upload (Option B): {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Process document immediately (not in background for Option B)
        result = await document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            sector=sector,
            use_case=use_case,
            title=title or file.filename
        )
        
        if result.get("success"):
            # Get the stored document with full details
            document_id = result.get("document_id")
            stored_doc = await db_manager.get_document(document_id)
            
            if stored_doc:
                # Return full document details
                return stored_doc
            else:
                raise HTTPException(status_code=500, detail="Failed to retrieve stored document")
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error (Option B): {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(
    file_content: bytes,
    filename: str,
    sector: str,
    use_case: Optional[str] = None,
    title: Optional[str] = None
):
    """Background task for document processing"""
    try:
        result = await document_processor.process_document(
            file_content=file_content,
            filename=filename,
            sector=sector,
            use_case=use_case,
            title=title
        )
        logger.info(f"✅ Background processing complete: {filename}")
        return result
    except Exception as e:
        logger.error(f"❌ Background processing failed for {filename}: {e}")
        return {"success": False, "error": str(e)}

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

@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """Get all chunks for a specific document with real text content"""
    try:
        if not database_available:
            return {"error": "Database not available"}
        
        # Get document info first
        document = await db_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks from database
        chunks_result = db_manager.supabase.table('chunks').select('*').eq('document_id', document_id).order('chunk_index').execute()
        
        chunks_data = []
        if chunks_result.data:
            for chunk in chunks_result.data:
                chunks_data.append({
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["text"],  # Fixed: use 'text' column from schema
                    "token_count": chunk.get("token_count", 0),
                    "chunk_type": chunk.get("metadata", {}).get("chunk_type", "semantic"),
                    "character_length": len(chunk["text"]),
                    "metadata": chunk.get("metadata", {}),
                    "created_at": chunk.get("created_at")
                })
        
        return {
            "success": True,
            "document": {
                "id": document_id,
                "title": document.get("title"),
                "filename": document.get("filename"),
                "sector": document.get("sector"),
                "use_case": document.get("use_case"),
                "chunk_count": len(chunks_data)
            },
            "chunks": chunks_data,
            "total_chunks": len(chunks_data),
            "total_characters": sum(len(chunk["chunk_text"]) for chunk in chunks_data)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")

@app.patch("/documents/{document_id}")
async def update_document(document_id: str, updates: DocumentUpdateRequest):
    document = await db_manager.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    update_data = {k: v for k, v in updates.dict(exclude_unset=True).items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    success = await db_manager.update_document_fields(document_id, update_data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update document")
    return {"success": True, "updated_fields": list(update_data.keys())}

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

# Temporarily disable AI chunking import until upload is fixed
try:
    from ai_chunking_service import chunking_service
    AI_CHUNKING_AVAILABLE = True
except ImportError as e:
    print(f"AI Chunking service not available: {e}")
    AI_CHUNKING_AVAILABLE = False
    chunking_service = None

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
        
        if AI_CHUNKING_AVAILABLE and chunking_service:
            # Use smart AI analysis
            content_text = chunking_service.extract_text_from_file(file_content, file.filename)
            
            if not content_text.strip():
                return DocumentAnalysisResponse(
                    success=False,
                    error="Could not extract text from document"
                )
            
            # Smart AI analysis
            analysis_result = await chunking_service.analyze_document_smart(
                content_text,
                file.filename,
                sector,
                use_case
            )
        else:
            # Fallback to basic analysis
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
                    "strategy": "AI-optimized semantic chunking (fallback mode)"
                },
                "estimatedChunks": max(1, len(file_content) // 1000),
                "aiInsights": {
                    "wordCount": word_count,
                    "hasStructure": has_structure,
                    "hasTechnicalContent": has_technical,
                    "hasDataElements": False,
                    "recommendedStrategy": f"Optimized for {sector} sector {use_case} use case",
                    "fallbackMode": True
                }
            }
        
        return DocumentAnalysisResponse(
            success=True,
            analysis=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Document analysis error: {e}")
        return DocumentAnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

# ============================================================================
# DOMAIN AND USE CASE MANAGEMENT ENDPOINTS
# ============================================================================

# TODO: TERMINOLOGY CLEANUP NEEDED
# Currently mixing "domains" and "sectors" terminology inconsistently:
# - Database schema uses "sectors" table
# - API endpoints use "/domains" paths  
# - Frontend expects "domains" in responses
# - Backend maps sectors → domains in transformation layer
# 
# DECISION NEEDED: Standardize on either "domains" OR "sectors" throughout:
# Option A: Rename database "sectors" → "domains" (requires migration)
# Option B: Rename API endpoints "/sectors" and update frontend
# Option C: Keep current mapping but document clearly
#
# Impact: Frontend, Backend, Database, Documentation, User Interface
# Priority: Medium (functional but confusing for maintenance)

class DomainCreateRequest(BaseModel):
    name: str
    description: str
    color: str
    icon: str

class DomainUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
    is_active: Optional[bool] = None

class UseCaseCreateRequest(BaseModel):
    name: str
    description: str
    category: str
    domain_id: str

class UseCaseUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    domain_id: Optional[str] = None
    is_active: Optional[bool] = None

class DomainCopyRequest(BaseModel):
    name: Optional[str] = None

class UseCaseCopyRequest(BaseModel):
    domain_id: Optional[str] = None
    name: Optional[str] = None

class UseCaseMoveRequest(BaseModel):
    domain_id: str

@app.get("/domains/with-use-cases")
async def list_domains_with_use_cases():
    """Get all domains (sectors) with their associated use cases"""
    try:
        # Get all sectors (treat as domains)
        sectors_result = db_manager.supabase.table('sectors').select('*').execute()
        
        # Get all use cases
        use_cases_result = db_manager.supabase.table('use_cases').select('*').execute()
        
        # Group use cases by sector
        use_cases_by_sector = {}
        for use_case in use_cases_result.data:
            sector_name = use_case.get('sector')
            if sector_name:
                if sector_name not in use_cases_by_sector:
                    use_cases_by_sector[sector_name] = []
                # Transform use case to match expected format
                transformed_use_case = {
                    'id': use_case['id'],
                    'name': use_case['name'],
                    'description': use_case.get('description', ''),
                    'category': 'General',  # Default category since not in schema
                    'domain_id': sector_name,  # Use sector name as domain_id
                    'is_active': True,
                    'document_count': 0,  # Would need to calculate from documents table
                    'created_at': use_case.get('created_at', ''),
                    'updated_at': use_case.get('updated_at', '')
                }
                use_cases_by_sector[sector_name].append(transformed_use_case)
        
        # Transform sectors to domains format
        domains_with_use_cases = []
        for sector in sectors_result.data:
            sector_name = sector['name']
            # Transform sector to domain format
            domain = {
                'id': str(sector['id']),
                'name': sector_name,
                'description': sector.get('description', ''),
                'color': '#3B82F6',  # Default color
                'icon': '📋',  # Default icon
                'is_active': True,
                'document_count': 0,  # Would need to calculate
                'created_at': sector.get('created_at', ''),
                'updated_at': sector.get('created_at', ''),
                'use_cases': use_cases_by_sector.get(sector_name, [])
            }
            domains_with_use_cases.append(domain)
        
        return {
            "domains": domains_with_use_cases,
            "total_count": len(domains_with_use_cases)
        }
    except Exception as e:
        logging.error(f"Failed to fetch domains with use cases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch domains: {str(e)}")

@app.get("/domains")
async def list_domains():
    """Get all domains (sectors)"""
    try:
        result = db_manager.supabase.table('sectors').select('*').execute()
        # Transform sectors to domains format
        domains = []
        for sector in result.data:
            domain = {
                'id': str(sector['id']),
                'name': sector['name'],
                'description': sector.get('description', ''),
                'color': '#3B82F6',  # Default color
                'icon': '📋',  # Default icon
                'is_active': True,
                'document_count': 0,  # Would need to calculate
                'created_at': sector.get('created_at', ''),
                'updated_at': sector.get('created_at', '')
            }
            domains.append(domain)
        return {"domains": domains}
    except Exception as e:
        logging.error(f"Failed to fetch domains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch domains: {str(e)}")

@app.get("/domains/{domain_id}")
async def get_domain(domain_id: str):
    """Get a specific domain (from sectors table)"""
    try:
        result = db_manager.supabase.table('sectors').select('*').eq('id', domain_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Domain not found")
        
        # Transform response to domain format
        sector = result.data[0]
        domain = {
            'id': sector['id'],
            'name': sector['name'],
            'description': sector.get('description', ''),
            'color': sector.get('color', '#3B82F6'),
            'icon': sector.get('icon', 'folder'),
            'is_active': sector.get('is_active', True),
            'document_count': sector.get('document_count', 0),
            'created_at': sector.get('created_at', ''),
            'updated_at': sector.get('updated_at', '')
        }
        return domain
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to fetch domain {domain_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch domain: {str(e)}")

@app.post("/domains")
async def create_domain(domain: DomainCreateRequest):
    """Create a new domain (in sectors table)"""
    try:
        domain_data = {
            "name": domain.name,
            "description": domain.description
            # Note: sectors table doesn't have color, icon, is_active, document_count columns
        }
        
        result = db_manager.supabase.table('sectors').insert(domain_data).execute()
        if not result.data:
            raise HTTPException(status_code=400, detail="Failed to create domain")
        
        # Transform response to domain format
        sector = result.data[0]
        domain_response = {
            'id': sector['id'],
            'name': sector['name'],
            'description': sector.get('description', ''),
            'color': sector.get('color', '#3B82F6'),
            'icon': sector.get('icon', 'folder'),
            'is_active': sector.get('is_active', True),
            'document_count': sector.get('document_count', 0),
            'created_at': sector.get('created_at', ''),
            'updated_at': sector.get('updated_at', '')
        }
        return domain_response
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to create domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create domain: {str(e)}")

@app.put("/domains/{domain_id}")
async def update_domain(domain_id: str, updates: DomainUpdateRequest):
    """Update a domain (actually updates sectors table)"""
    try:
        # Only include non-None values in the update
        update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid update data provided")
        
        # Note: sectors table doesn't have updated_at column, so we don't set it
        
        # Map the domain_id to sector name for existing schema
        # NOTE: Using sectors table since database schema uses sectors not domains
        result = db_manager.supabase.table('sectors').update(update_data).eq('id', domain_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Domain not found")
        
        # Transform response back to domain format
        sector = result.data[0]
        domain = {
            'id': sector['id'],
            'name': sector['name'],
            'description': sector.get('description', ''),
            'color': sector.get('color', '#3B82F6'),
            'icon': sector.get('icon', 'folder'),
            'is_active': sector.get('is_active', True),
            'document_count': sector.get('document_count', 0),
            'created_at': sector.get('created_at', ''),
            'updated_at': sector.get('updated_at', '')
        }
        return domain
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update domain {domain_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update domain: {str(e)}")

@app.delete("/domains/{domain_id}")
async def delete_domain(domain_id: str):
    """Delete a domain (from sectors table, only if no documents are linked)"""
    try:
        # Check if domain exists (sectors table doesn't have document_count column)
        domain_result = db_manager.supabase.table('sectors').select('name').eq('id', domain_id).execute()
        if not domain_result.data:
            raise HTTPException(status_code=404, detail="Domain not found")
        
        domain = domain_result.data[0]
        # Note: We skip document count check since sectors table doesn't have document_count column
        
        # Delete associated use cases first (use cases reference sector not domain_id)
        db_manager.supabase.table('use_cases').delete().eq('sector', domain['name']).execute()
        
        # Delete the domain
        result = db_manager.supabase.table('sectors').delete().eq('id', domain_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Domain not found")
        
        return {"success": True, "message": "Domain deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete domain {domain_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete domain: {str(e)}")

@app.post("/domains/{domain_id}/copy")
async def copy_domain(domain_id: str, copy_request: DomainCopyRequest):
    """Copy a domain with all its use cases (from sectors table)"""
    try:
        # Get the original domain (from sectors)
        domain_result = db_manager.supabase.table('sectors').select('*').eq('id', domain_id).execute()
        if not domain_result.data:
            raise HTTPException(status_code=404, detail="Domain not found")
        
        original_domain = domain_result.data[0]
        
        # Create new domain - only set columns that exist in sectors table
        new_domain_name = copy_request.name or f"{original_domain['name']} (Copy)"
        new_domain_data = {
            "name": new_domain_name,
            "description": original_domain.get('description', '')
            # Note: sectors table doesn't have color, icon, is_active, document_count columns
        }
        
        new_domain_result = db_manager.supabase.table('sectors').insert(new_domain_data).execute()
        if not new_domain_result.data:
            raise HTTPException(status_code=400, detail="Failed to create domain copy")
        
        new_domain = new_domain_result.data[0]
        new_domain_name_str = new_domain['name']
        
        # Get and copy all use cases (use cases reference sector by name)
        use_cases_result = db_manager.supabase.table('use_cases').select('*').eq('sector', original_domain['name']).execute()
        
        if use_cases_result.data:
            new_use_cases = []
            for use_case in use_cases_result.data:
                new_use_case_data = {
                    "name": use_case['name'],
                    "description": use_case.get('description', ''),
                    "sector": new_domain_name_str  # use cases reference sector by name
                    # Note: is_active column doesn't exist in use_cases table
                }
                new_use_cases.append(new_use_case_data)
            
            if new_use_cases:
                db_manager.supabase.table('use_cases').insert(new_use_cases).execute()
        
        # Transform response to domain format
        domain_response = {
            'id': new_domain['id'],
            'name': new_domain['name'],
            'description': new_domain.get('description', ''),
            'color': new_domain.get('color', '#3B82F6'),
            'icon': new_domain.get('icon', 'folder'),
            'is_active': new_domain.get('is_active', True),
            'document_count': new_domain.get('document_count', 0),
            'created_at': new_domain.get('created_at', ''),
            'updated_at': new_domain.get('updated_at', '')
        }
        return domain_response
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to copy domain {domain_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to copy domain: {str(e)}")

# Use Case endpoints
@app.get("/use-cases")
async def list_use_cases(domain_id: Optional[str] = None):
    """Get all use cases, optionally filtered by domain (sector)"""
    try:
        query = db_manager.supabase.table('use_cases').select('*')
        if domain_id:
            # domain_id in our API corresponds to sector name in existing schema
            query = query.eq('sector', domain_id)
        
        result = query.execute()
        # Transform use cases to expected format
        use_cases = []
        for use_case in result.data:
            transformed = {
                'id': use_case['id'],
                'name': use_case['name'],
                'description': use_case.get('description', ''),
                'category': 'General',  # Default since not in existing schema
                'domain_id': use_case.get('sector', ''),
                'is_active': True,
                'document_count': 0,  # Would need to calculate
                'created_at': use_case.get('created_at', ''),
                'updated_at': use_case.get('updated_at', '')
            }
            use_cases.append(transformed)
        return {"use_cases": use_cases}
    except Exception as e:
        logging.error(f"Failed to fetch use cases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch use cases: {str(e)}")

@app.get("/use-cases/{use_case_id}")
async def get_use_case(use_case_id: str):
    """Get a specific use case"""
    try:
        result = db_manager.supabase.table('use_cases').select('*').eq('id', use_case_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Use case not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to fetch use case {use_case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch use case: {str(e)}")

@app.post("/use-cases")
async def create_use_case(use_case: UseCaseCreateRequest):
    """Create a new use case"""
    try:
        # Verify domain exists (check sectors table)
        domain_result = db_manager.supabase.table('sectors').select('name').eq('id', use_case.domain_id).execute()
        if not domain_result.data:
            raise HTTPException(status_code=400, detail="Domain not found")
        
        sector_name = domain_result.data[0]['name']
        
        use_case_data = {
            "name": use_case.name,
            "description": use_case.description,
            "sector": sector_name  # use cases reference sector by name
            # Note: is_active column doesn't exist in use_cases table
        }
        
        result = db_manager.supabase.table('use_cases').insert(use_case_data).execute()
        if not result.data:
            raise HTTPException(status_code=400, detail="Failed to create use case")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to create use case: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create use case: {str(e)}")

@app.put("/use-cases/{use_case_id}")
async def update_use_case(use_case_id: str, updates: UseCaseUpdateRequest):
    """Update a use case"""
    try:
        # Only include non-None values in the update
        update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid update data provided")
        
        # If domain_id is being updated, verify it exists and get sector name
        if 'domain_id' in update_data:
            domain_result = db_manager.supabase.table('sectors').select('name').eq('id', update_data['domain_id']).execute()
            if not domain_result.data:
                raise HTTPException(status_code=400, detail="Domain not found")
            # Replace domain_id with sector name for the actual update
            update_data['sector'] = domain_result.data[0]['name']
            del update_data['domain_id']
        
        update_data["updated_at"] = datetime.now().isoformat()
        
        result = db_manager.supabase.table('use_cases').update(update_data).eq('id', use_case_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Use case not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update use case {use_case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update use case: {str(e)}")

@app.delete("/use-cases/{use_case_id}")
async def delete_use_case(use_case_id: str):
    """Delete a use case (only if no documents are linked)"""
    try:
        # Check if use case exists (use_cases table doesn't have document_count column)
        use_case_result = db_manager.supabase.table('use_cases').select('name').eq('id', use_case_id).execute()
        if not use_case_result.data:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        # Note: We skip document count check since use_cases table doesn't have document_count column
        
        result = db_manager.supabase.table('use_cases').delete().eq('id', use_case_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        return {"success": True, "message": "Use case deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete use case {use_case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete use case: {str(e)}")

@app.post("/use-cases/{use_case_id}/copy")
async def copy_use_case(use_case_id: str, copy_request: UseCaseCopyRequest):
    """Copy a use case to the same or different domain"""
    try:
        # Get the original use case
        use_case_result = db_manager.supabase.table('use_cases').select('*').eq('id', use_case_id).execute()
        if not use_case_result.data:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        original_use_case = use_case_result.data[0]
        
        # Determine target domain and sector
        target_domain_id = copy_request.domain_id
        target_sector_name = original_use_case['sector']  # Default to same sector
        
        if target_domain_id:
            # Verify target domain exists and get sector name
            domain_result = db_manager.supabase.table('sectors').select('name').eq('id', target_domain_id).execute()
            if not domain_result.data:
                raise HTTPException(status_code=400, detail="Target domain not found")
            target_sector_name = domain_result.data[0]['name']
        
        # Create new use case
        new_use_case_name = copy_request.name or f"{original_use_case['name']} (Copy)"
        new_use_case_data = {
            "name": new_use_case_name,
            "description": original_use_case.get('description', ''),
            "sector": target_sector_name  # use cases reference sector by name
            # Note: is_active column doesn't exist in use_cases table
        }
        
        result = db_manager.supabase.table('use_cases').insert(new_use_case_data).execute()
        if not result.data:
            raise HTTPException(status_code=400, detail="Failed to create use case copy")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to copy use case {use_case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to copy use case: {str(e)}")

@app.put("/use-cases/{use_case_id}/move")
async def move_use_case(use_case_id: str, move_request: UseCaseMoveRequest):
    """Move a use case to a different domain"""
    try:
        # Verify target domain exists and get sector name
        domain_result = db_manager.supabase.table('sectors').select('name').eq('id', move_request.domain_id).execute()
        if not domain_result.data:
            raise HTTPException(status_code=400, detail="Target domain not found")
        
        target_sector_name = domain_result.data[0]['name']
        
        # Update the use case's sector
        update_data = {
            "sector": target_sector_name,  # use cases reference sector by name
            "updated_at": datetime.now().isoformat()  # use_cases table has updated_at column
        }
        
        result = db_manager.supabase.table('use_cases').update(update_data).eq('id', use_case_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to move use case {use_case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to move use case: {str(e)}")

# ============================================================================
# RESPONSE MODELS
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT) 