# main.py - Main FastAPI Application (WITH FEEDBACK SYSTEM)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from typing import List, Optional
import uvicorn
from datetime import datetime
import uuid
import os
import io
import csv
from pathlib import Path
import json
from websocket_manager import WebSocketManager
from fastapi_utils.tasks import repeat_every

# Import our modules
from database import DatabaseManager
from document_processor import EnhancedDocumentProcessor
from vector_store import PineconeManager
from web_scraper import ComprehensiveWebScraper
from ai_services import AIService
from auth import AdminAuth
from models import *  # All Pydantic models (now includes feedback models)
from config import settings

# Import enhanced modules
from specialized_agents import OrchestrationAgent
from report_generator import ReportGenerator, ReportTemplateManager
from map_data_manager import RailwayMapDataManager

# Initialize FastAPI app
app = FastAPI(
    title="Strategy AI Backend",
    description="AI Agent for Strategy Document Analysis with Admin Management & User Feedback",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for V0 frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all services
db_manager = DatabaseManager()
doc_processor = EnhancedDocumentProcessor()
pinecone_manager = PineconeManager()
web_scraper = ComprehensiveWebScraper()
ai_service = AIService()
admin_auth = AdminAuth()

# Initialize enhanced services
orchestration_agent = OrchestrationAgent()
report_generator = ReportGenerator()
template_manager = ReportTemplateManager()
map_data_manager = RailwayMapDataManager()

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "message": "Strategy AI Backend is running"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    try:
        db_status = await db_manager.test_connection()
        pinecone_status = await pinecone_manager.test_connection()
        
        return {
            "status": "healthy" if db_status and pinecone_status else "degraded",
            "services": {
                "database": "connected" if db_status else "error",
                "vector_store": "connected" if pinecone_status else "error",
                "document_processor": "ready",
                "web_scraper": "ready",
                "ai_service": "ready",
                "feedback_system": "enabled"
            },
            "metrics": {
                "total_documents": await db_manager.get_document_count(),
                "total_sectors": await db_manager.get_sector_count(),
                "total_use_cases": await db_manager.get_use_case_count(),
                "total_feedback": await db_manager.get_feedback_count()
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ============================================================================
# MAIN CHAT ENDPOINT (For V0 Frontend)
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """Main chat endpoint for V0 frontend"""
    try:
        # Auto-detect use case if not specified
        if not message.use_case:
            message.use_case = await ai_service.detect_use_case(
                message.message, message.sector
            )

        # Get custom prompt template
        prompt_template = await db_manager.get_prompt_template(
            message.sector, message.use_case
        )

        # Retrieve relevant documents
        relevant_docs = await pinecone_manager.semantic_search(
            query=message.message,
            filters={"sector": message.sector, "use_case": message.use_case},
            top_k=8
        )

        # Generate AI response
        ai_response = await ai_service.generate_response(
            query=message.message,
            context_docs=relevant_docs,
            prompt_template=prompt_template,
            user_type=message.user_type
        )

        # Format sources
        formatted_sources = [
            {
                "document_title": doc.get("metadata", {}).get("title", "Untitled"),
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "relevance_score": doc.get("score", 0.0),
                "chunk_preview": doc.get("text", "")[:200] + "..."
            }
            for doc in relevant_docs[:5]
        ]

        # Log interaction and get chat_log_id for feedback
        chat_log_id = await db_manager.log_chat_interaction(
            session_id=message.session_id,
            query=message.message,
            response=ai_response,
            sources_used=[doc.get("id") for doc in relevant_docs],
            sector=message.sector,
            use_case=message.use_case
        )

        return ChatResponse(
            response=ai_response,
            sources=formatted_sources,
            confidence=ai_service.calculate_confidence(relevant_docs),
            suggested_use_case=message.use_case,
            timestamp=datetime.now(),
            chat_log_id=chat_log_id  # Include for feedback linking
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/chat/advanced", response_model=ChatResponse)
async def advanced_chat_with_ai(message: ChatMessage):
    """
    Advanced chat endpoint using specialized AI agents
    Provides better responses through agent orchestration
    """
    try:
        # Determine complexity and agent requirements
        complexity = "comprehensive" if len(message.message) > 100 else "simple"
        
        # Retrieve relevant documents from vector store
        relevant_docs = await pinecone_manager.semantic_search(
            query=message.message,
            filters={"sector": message.sector, "use_case": message.use_case},
            top_k=8
        )

        # Prepare request for orchestration agent
        agent_request = {
            "type": "chat" if complexity == "simple" else "analysis",
            "query": message.message,
            "context": _prepare_context_for_agents(relevant_docs),
            "sector": message.sector,
            "use_case": message.use_case,
            "user_type": message.user_type,
            "complexity": complexity,
            "conversational_format": True
        }

        # Process with specialized agents
        agent_response = await orchestration_agent.process_request(agent_request)

        # Extract primary response
        if "primary_response" in agent_response:
            ai_response = agent_response["primary_response"]
            agents_used = agent_response.get("agents_used", [])
        else:
            ai_response = agent_response.get("response", "I'm having trouble processing your request.")
            agents_used = [agent_response.get("agent", "Unknown")]

        # Format sources for frontend
        formatted_sources = [
            {
                "document_title": doc.get("metadata", {}).get("title", "Untitled"),
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "relevance_score": doc.get("score", 0.0),
                "chunk_preview": doc.get("text", "")[:200] + "..."
            }
            for doc in relevant_docs[:5]
        ]

        # Log the enhanced interaction
        chat_log_id = await db_manager.log_chat_interaction(
            session_id=message.session_id,
            query=message.message,
            response=ai_response,
            sources_used=[doc.get("id") for doc in relevant_docs],
            sector=message.sector,
            use_case=message.use_case
        )

        return ChatResponse(
            response=ai_response,
            sources=formatted_sources,
            confidence=agent_response.get("confidence", 0.8),
            suggested_use_case=message.use_case,
            timestamp=datetime.now(),
            chat_log_id=chat_log_id,  # Include for feedback linking
            enhanced_features={
                "agents_used": agents_used,
                "response_type": agent_response.get("response_type", "standard"),
                "analysis_available": "detailed_analysis" in agent_response
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in advanced chat: {str(e)}")

def _prepare_context_for_agents(docs: List[Dict]) -> str:
    """Prepare context string optimized for AI agents"""
    if not docs:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(docs[:5]):
        doc_title = doc.get("metadata", {}).get("title", f"Document {i+1}")
        doc_text = doc.get("text", "")
        doc_sector = doc.get("metadata", {}).get("sector", "")
        
        context_parts.append(f"Source {i+1} - {doc_title} ({doc_sector}):\n{doc_text}\n")
    
    return "\n".join(context_parts)

@app.get("/api/chat/suggest-use-case")
async def suggest_use_case(query: str, sector: str = "General"):
    """
    Conversational guidance feature - suggests best use case from user description
    For the hybrid dropdown + conversational interface
    """
    try:
        suggested_use_case = await ai_service.detect_use_case(query, sector)
        use_case_info = await db_manager.get_use_case_info(sector, suggested_use_case)
        
        return {
            "suggested_use_case": suggested_use_case,
            "confidence": 0.8,
            "description": use_case_info.get("description", ""),
            "example_queries": use_case_info.get("example_queries", []),
            "explanation": f"Based on your query, '{suggested_use_case}' seems most relevant for {sector} sector"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error suggesting use case: {str(e)}")

# ============================================================================
# USER FEEDBACK ENDPOINTS (NEW)
# ============================================================================

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: UserFeedback, request: Request):
    """Submit user feedback on AI responses or documents"""
    try:
        feedback_id = str(uuid.uuid4())
        
        # Get client info for analytics
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        feedback_data = {
            "id": feedback_id,
            "chat_log_id": feedback.chat_log_id,
            "document_id": feedback.document_id,
            "session_id": feedback.session_id,
            "rating": feedback.rating,
            "feedback_type": feedback.feedback_type,
            "comment": feedback.comment,
            "helpful": feedback.helpful,
            "user_agent": user_agent,
            "ip_address": client_host,
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db_manager.store_feedback(feedback_data)
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Thank you for your feedback! Your input helps us improve."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {str(e)}")

@app.get("/api/feedback/analytics", response_model=FeedbackAnalytics)
async def get_feedback_analytics(
    days: int = 30,
    feedback_type: Optional[str] = None,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Get feedback analytics for admin dashboard"""
    try:
        analytics = await db_manager.get_feedback_analytics(days=days, feedback_type=feedback_type)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback analytics: {str(e)}")

@app.get("/api/feedback/recent")
async def get_recent_feedback(
    limit: int = 20,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Get recent feedback for monitoring"""
    try:
        recent_feedback = await db_manager.get_recent_feedback(limit=limit)
        return {"recent_feedback": recent_feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent feedback: {str(e)}")

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS (Enhanced with Feedback)
# ============================================================================

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sector: str = Form("General"),
    use_case: str = Form(None),
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Upload and process document with real-time status updates"""
    try:
        document_id = str(uuid.uuid4())
        
        # Save file
        file_path = f"uploads/{document_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Start processing in background
        background_tasks.add_task(
            process_document_with_updates,
            document_id,
            file_path,
            sector,
            use_case
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document upload started. Connect to WebSocket for progress updates."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    sector: Optional[str] = None,
    use_case: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    min_rating: Optional[float] = None,  # New: Filter by minimum rating
    limit: int = 50,
    offset: int = 0
):
    """List documents with filtering (enhanced with feedback data)"""
    try:
        documents = await db_manager.get_documents(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search,
            min_rating=min_rating,
            limit=limit,
            offset=offset
        )

        total_count = await db_manager.get_document_count(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search,
            min_rating=min_rating
        )

        return DocumentListResponse(
            documents=documents,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

# ============================================================================
# REPORT GENERATION ENDPOINTS
# ============================================================================

@app.post("/api/reports/generate")
async def generate_report(
    background_tasks: BackgroundTasks,
    report_type: str = Form(...),
    sector: str = Form("General"),
    format: str = Form("pdf"),  # pdf, docx, both
    title: Optional[str] = Form(None),
    scope: str = Form("comprehensive"),
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """
    Generate comprehensive reports using AI agents
    Supports PDF and DOCX formats with professional styling
    """
    try:
        # Validate report type
        available_types = [rt["type"] for rt in report_generator.get_available_report_types()]
        if report_type not in available_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report type. Available: {available_types}"
            )

        # Prepare report parameters
        parameters = {
            "sector": sector,
            "title": title,
            "scope": scope,
            "generated_by": "admin",
            "request_timestamp": datetime.now().isoformat()
        }

        # Start report generation in background
        report_task = report_generator.generate_report(
            report_type=report_type,
            parameters=parameters,
            format=format
        )

        # Execute the report generation
        result = await report_task

        if result["success"]:
            return {
                "success": True,
                "report_id": result["report_id"],
                "download_urls": result["download_urls"],
                "metadata": result["metadata"],
                "message": f"Report generated successfully in {format} format"
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/api/reports/{report_id}/download/{filename}")
async def download_report(report_id: str, filename: str):
    """Download generated report file"""
    try:
        file_path = await report_generator.get_report_file(report_id, filename)
        
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        # Determine content type
        if filename.endswith('.pdf'):
            media_type = 'application/pdf'
        elif filename.endswith('.docx'):
            media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            media_type = 'application/octet-stream'

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading report: {str(e)}")

@app.post("/api/reports/generate-stream")
async def generate_report_stream(
    report_type: str = Form(...),
    sector: str = Form("General"),
    format: str = Form("pdf"),
    title: Optional[str] = Form(None),
    scope: str = Form("comprehensive"),
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """
    Generate report with streaming response
    Sends each section as it's generated
    """
    async def generate():
        try:
            # Validate report type
            available_types = [rt["type"] for rt in report_generator.get_available_report_types()]
            if report_type not in available_types:
                yield f"data: {json.dumps({'error': f'Invalid report type. Available: {available_types}'})}\n\n"
                return

            # Prepare report parameters
            parameters = {
                "sector": sector,
                "title": title,
                "scope": scope,
                "generated_by": "admin",
                "request_timestamp": datetime.now().isoformat()
            }

            # Get report template sections
            template = report_generator.report_templates.get(report_type, {})
            sections = template.get("sections", [])

            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Initializing report generation'})}\n\n"

            # Generate each section
            report_content = {}
            for i, section in enumerate(sections, 1):
                try:
                    # Update status
                    progress = int((i / len(sections)) * 100)
                    yield f"data: {json.dumps({'status': 'generating', 'section': section, 'progress': progress})}\n\n"

                    # Generate section content
                    section_content = await report_generator.orchestrator.generate_section(
                        section=section,
                        report_type=report_type,
                        parameters=parameters
                    )

                    report_content[section] = section_content
                    
                    # Send section content
                    message = {
                        'status': 'section_complete',
                        'section': section,
                        'content': section_content,
                        'progress': progress
                    }
                    yield f"data: {json.dumps(message)}\n\n"

                except Exception as e:
                    error_message = {'error': f'Error generating section {section}: {str(e)}'}
                    yield f"data: {json.dumps(error_message)}\n\n"

            # Generate final documents
            try:
                report_id = str(uuid.uuid4())
                
                # Create PDF/DOCX
                generated_files = []
                if format in ["pdf", "both"]:
                    pdf_path = await report_generator._generate_pdf_report(
                        report_id, report_type, {"sections": report_content}, parameters
                    )
                    if pdf_path:
                        generated_files.append({
                            "format": "pdf",
                            "path": pdf_path,
                            "filename": f"{report_type}_{report_id}.pdf"
                        })

                if format in ["docx", "both"]:
                    docx_path = await report_generator._generate_docx_report(
                        report_id, report_type, {"sections": report_content}, parameters
                    )
                    if docx_path:
                        generated_files.append({
                            "format": "docx",
                            "path": docx_path,
                            "filename": f"{report_type}_{report_id}.docx"
                        })

                # Save metadata
                metadata = {
                    "report_id": report_id,
                    "report_type": report_type,
                    "generated_at": datetime.now().isoformat(),
                    "parameters": parameters,
                    "files": generated_files
                }
                await report_generator._save_report_metadata(report_id, metadata)

                # Send completion status with download URLs
                completion_message = {
                    'status': 'complete',
                    'report_id': report_id,
                    'download_urls': [f"/api/reports/{report_id}/download/{file['filename']}" for file in generated_files]
                }
                yield f"data: {json.dumps(completion_message)}\n\n"

            except Exception as e:
                error_message = {'error': f'Error finalizing report: {str(e)}'}
                yield f"data: {json.dumps(error_message)}\n\n"

        except Exception as e:
            error_message = {'error': f'Error in report generation: {str(e)}'}
            yield f"data: {json.dumps(error_message)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================================
# RAILWAY MAP DATA ENDPOINTS
# ============================================================================

@app.get("/api/map/regions")
async def get_railway_regions():
    """Get railway regions GeoJSON data for map visualization"""
    try:
        regions_data = await map_data_manager.get_railway_regions_geojson()
        return regions_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting railway regions: {str(e)}")

@app.get("/api/map/lines")
async def get_railway_lines():
    """Get railway lines GeoJSON data"""
    try:
        lines_data = await map_data_manager.get_railway_lines_geojson()
        return lines_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting railway lines: {str(e)}")

@app.get("/api/map/stations")
async def get_railway_stations(region: Optional[str] = None):
    """Get railway stations GeoJSON data, optionally filtered by region"""
    try:
        stations_data = await map_data_manager.get_stations_geojson(region=region)
        return stations_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting railway stations: {str(e)}")

# ============================================================================
# ADMIN ANALYTICS ENDPOINTS (Enhanced with Feedback)
# ============================================================================

@app.get("/api/admin/analytics", response_model=EnhancedSystemAnalytics)
async def get_enhanced_system_analytics(admin_key: str = Depends(admin_auth.verify_admin)):
    """Get comprehensive system analytics including feedback data"""
    try:
        # Get base analytics
        base_analytics = await db_manager.get_system_analytics()
        
        # Get feedback analytics
        feedback_analytics = await db_manager.get_feedback_analytics()
        
        # Combine into enhanced analytics
        enhanced_analytics = EnhancedSystemAnalytics(
            **base_analytics,
            feedback_summary=feedback_analytics
        )
        
        return enhanced_analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

@app.get("/api/admin/active-processes")
async def get_active_processes(admin_key: str = Depends(admin_auth.verify_admin)):
    """Get all active document processing and report generation tasks"""
    try:
        # Get active document processes
        active_docs = websocket_manager.get_active_processes()
        
        # Get active report generations
        active_reports = {}
        for report_id in report_generator.cache.cache.keys():
            progress = await report_generator.get_report_progress(report_id)
            if progress:
                active_reports[report_id] = progress
        
        return {
            "active_documents": active_docs,
            "active_reports": active_reports,
            "total_active_processes": len(active_docs) + len(active_reports),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting active processes: {str(e)}")

@app.on_event("startup")
@repeat_every(seconds=60)
async def cleanup_stale_tasks():
    """Monitor and clean up stale processing tasks"""
    try:
        # Clean up stale WebSocket connections
        await websocket_manager.cleanup_stale_connections(max_age_minutes=30)
        
        # Clean up old reports
        await report_generator.cleanup_old_reports(max_age_hours=24)
        
        # Clean up documents stuck in processing
        stale_docs = await db_manager.get_stale_documents(
            max_age_minutes=30,
            status="processing"
        )
        
        for doc in stale_docs:
            await websocket_manager.send_error(
                doc["id"],
                "Processing timed out. Please try again.",
                status_code=408,
                details={"timeout_minutes": 30}
            )
            await db_manager.update_document_status(
                doc["id"],
                "error",
                error="Processing timeout"
            )
            
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}")

# ============================================================================
# SYSTEM ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/system/health")
async def get_system_health(admin_key: str = Depends(admin_auth.verify_admin)):
    """Get comprehensive system health analysis"""
    try:
        # Collect system metrics
        metrics = await collect_system_metrics()
        
        # Prepare request for system analytics agent
        analytics_request = {
            "type": "system_analytics",
            "analysis_type": "system_health",
            "time_range": "24h",
            "metrics": metrics
        }
        
        # Process with system analytics agent
        analytics_response = await orchestration_agent.process_request(analytics_request)
        
        return analytics_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing system health: {str(e)}")

@app.get("/api/system/performance")
async def get_system_performance(
    time_range: str = "24h",
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Get detailed system performance metrics and analysis"""
    try:
        # Collect performance metrics
        metrics = await collect_performance_metrics(time_range)
        
        analytics_request = {
            "type": "system_analytics",
            "analysis_type": "performance_metrics",
            "time_range": time_range,
            "metrics": metrics
        }
        
        analytics_response = await orchestration_agent.process_request(analytics_request)
        
        return analytics_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing system performance: {str(e)}")

@app.get("/api/system/feedback/analysis")
async def analyze_user_feedback(
    time_range: str = "7d",
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Analyze user feedback and satisfaction metrics"""
    try:
        # Collect user feedback data
        feedback_data = await collect_user_feedback(time_range)
        
        analytics_request = {
            "type": "system_analytics",
            "analysis_type": "user_feedback",
            "time_range": time_range,
            "feedback_data": feedback_data
        }
        
        analytics_response = await orchestration_agent.process_request(analytics_request)
        
        return analytics_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing user feedback: {str(e)}")

@app.get("/api/system/comprehensive")
async def get_comprehensive_analysis(
    time_range: str = "24h",
    include_feedback: bool = True,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Get comprehensive system analysis including health, performance, and feedback"""
    try:
        # Collect all relevant data
        metrics = await collect_system_metrics()
        performance_metrics = await collect_performance_metrics(time_range)
        feedback_data = await collect_user_feedback(time_range) if include_feedback else []
        
        analytics_request = {
            "type": "comprehensive",
            "metrics": {
                "system": metrics,
                "performance": performance_metrics
            },
            "feedback_data": feedback_data,
            "time_range": time_range
        }
        
        analytics_response = await orchestration_agent.process_request(analytics_request)
        
        return analytics_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing comprehensive analysis: {str(e)}")

# ============================================================================
# BACKGROUND PROCESSING FUNCTIONS
# ============================================================================

async def process_document_with_updates(
    document_id: str,
    file_path: str,
    sector: str,
    use_case: str
):
    """Process document with WebSocket status updates"""
    try:
        # Initial status
        await websocket_manager.send_processing_update(
            document_id,
            "started",
            progress=0,
            message="Starting document processing"
        )
        
        # Extract text
        await websocket_manager.send_processing_update(
            document_id,
            "extracting_text",
            progress=10,
            message="Extracting text from document"
        )
        
        text = await doc_processor.extract_text(file_path)
        
        # Generate chunks
        await websocket_manager.send_processing_update(
            document_id,
            "chunking",
            progress=30,
            message="Generating document chunks"
        )
        
        chunks = await doc_processor.generate_chunks(text)
        total_chunks = len(chunks)
        
        # Process chunks and generate embeddings
        for i, chunk in enumerate(chunks, 1):
            progress = 30 + int((i / total_chunks) * 40)  # Progress from 30% to 70%
            await websocket_manager.send_processing_update(
                document_id,
                "generating_embeddings",
                progress=progress,
                message=f"Generating embeddings for chunk {i}/{total_chunks}"
            )
            
            await pinecone_manager.add_chunk(
                document_id=document_id,
                chunk_text=chunk,
                metadata={
                    "sector": sector,
                    "use_case": use_case,
                    "chunk_number": i
                }
            )
        
        # Generate metadata
        await websocket_manager.send_processing_update(
            document_id,
            "generating_metadata",
            progress=80,
            message="Generating document metadata"
        )
        
        metadata = await doc_processor.generate_metadata(text)
        
        # Save to database
        await websocket_manager.send_processing_update(
            document_id,
            "saving",
            progress=90,
            message="Saving document information"
        )
        
        await db_manager.save_document({
            "id": document_id,
            "filename": os.path.basename(file_path),
            "sector": sector,
            "use_case": use_case,
            "metadata": metadata,
            "status": "processed",
            "created_at": datetime.now().isoformat()
        })
        
        # Complete
        await websocket_manager.send_processing_update(
            document_id,
            "completed",
            progress=100,
            message="Document processing completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        await websocket_manager.send_error(document_id, str(e))
        
        # Update document status in database
        await db_manager.update_document_status(document_id, "error", error=str(e))

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

@app.websocket("/ws/document/{document_id}")
async def websocket_endpoint(websocket: WebSocket, document_id: str):
    """WebSocket endpoint for document processing updates"""
    await websocket_manager.connect(websocket, document_id)
    try:
        # Send initial status if document exists
        doc_status = await db_manager.get_document_status(document_id)
        if doc_status:
            await websocket_manager.send_processing_update(
                document_id,
                doc_status["status"],
                message=doc_status.get("message", ""),
                progress=doc_status.get("progress", 0)
            )
        
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, document_id)
    except Exception as e:
        logger.error(f"WebSocket error for document {document_id}: {e}")
        await websocket_manager.send_error(document_id, str(e))
        websocket_manager.disconnect(websocket, document_id)