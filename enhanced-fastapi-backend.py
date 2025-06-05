# main.py - Main FastAPI Application
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import uvicorn
from datetime import datetime
import uuid

# Import our modules (like your Streamlit structure)
from database import DatabaseManager
from document_processor import EnhancedDocumentProcessor
from vector_store import PineconeManager
from web_scraper import ComprehensiveWebScraper
from ai_services import AIService
from auth import AdminAuth
from models import *  # Pydantic models
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Strategy AI Backend",
    description="AI Agent for Strategy Document Analysis with Admin Management",
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

# Import enhanced modules
from specialized_agents import OrchestrationAgent
from report_generator import ReportGenerator, ReportTemplateManager
from map_data_manager import RailwayMapDataManager

# Initialize enhanced services
orchestration_agent = OrchestrationAgent()
report_generator = ReportGenerator()
template_manager = ReportTemplateManager()
map_data_manager = RailwayMapDataManager()

# ============================================================================
# ENHANCED CHAT ENDPOINTS (Multi-Agent Support)
# ============================================================================

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
        await db_manager.log_chat_interaction(
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

@app.get("/api/reports")
async def list_reports(
    limit: int = 50,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """List all generated reports with metadata"""
    try:
        reports = await report_generator.list_reports(limit=limit)
        return {"reports": reports, "total_count": len(reports)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")

@app.get("/api/reports/{report_id}/metadata")
async def get_report_metadata(report_id: str):
    """Get metadata for a specific report"""
    try:
        metadata = await report_generator.get_report_metadata(report_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return metadata

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting report metadata: {str(e)}")

@app.delete("/api/reports/{report_id}")
async def delete_report(
    report_id: str,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Delete a report and its files"""
    try:
        success = await report_generator.delete_report(report_id)
        
        if success:
            return {"success": True, "message": f"Report {report_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Report not found or could not be deleted")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting report: {str(e)}")

@app.get("/api/reports/types")
async def get_available_report_types():
    """Get information about available report types"""
    try:
        return {"report_types": report_generator.get_available_report_types()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting report types: {str(e)}")

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

@app.get("/api/map/projects/{region_id}")
async def get_projects_by_region(region_id: str):
    """Get projects associated with a specific railway region"""
    try:
        projects = await map_data_manager.get_projects_by_region(region_id)
        return {"region_id": region_id, "projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting projects for region: {str(e)}")

@app.get("/api/map/search")
async def search_map_locations(query: str):
    """Search for railway-related locations"""
    try:
        results = await map_data_manager.search_locations(query)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching locations: {str(e)}")

@app.get("/api/map/route")
async def get_route_between_stations(start: str, end: str):
    """Get route information between two railway stations"""
    try:
        route_data = await map_data_manager.get_route_between_stations(start, end)
        
        if not route_data:
            raise HTTPException(status_code=404, detail="Could not find route between specified stations")
        
        return route_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting route: {str(e)}")

@app.get("/api/map/config")
async def get_map_configuration():
    """Get map configuration for frontend visualization"""
    try:
        config = map_data_manager.get_map_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting map config: {str(e)}")

@app.get("/api/map/statistics")
async def get_map_statistics():
    """Get statistics about railway map data"""
    try:
        stats = await map_data_manager.get_map_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting map statistics: {str(e)}")

@app.post("/api/map/update")
async def update_map_data(admin_key: str = Depends(admin_auth.verify_admin)):
    """Update all map data (admin function)"""
    try:
        result = await map_data_manager.update_map_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating map data: {str(e)}")

# ============================================================================
# ENHANCED ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/admin/agents/status")
async def get_ai_agents_status(admin_key: str = Depends(admin_auth.verify_admin)):
    """Get status and information about available AI agents"""
    try:
        agents_info = orchestration_agent.get_available_agents()
        
        # Test each agent
        agent_status = []
        for agent_info in agents_info:
            try:
                # Simple test request
                test_request = {
                    "type": "chat",
                    "query": "test",
                    "context": "test context"
                }
                
                if agent_info["name"] == "ChatAgent":
                    test_response = await orchestration_agent.chat_agent.process(test_request)
                elif agent_info["name"] == "AnalysisAgent":
                    test_response = await orchestration_agent.analysis_agent.process(test_request)
                elif agent_info["name"] == "ReportAgent":
                    test_response = await orchestration_agent.report_agent.process(test_request)
                elif agent_info["name"] == "MetadataAgent":
                    test_response = await orchestration_agent.metadata_agent.process(test_request)
                
                status = "healthy" if test_response.get("confidence", 0) > 0 else "degraded"
                
            except Exception as e:
                status = "error"
            
            agent_status.append({
                **agent_info,
                "status": status,
                "last_tested": datetime.now().isoformat()
            })
        
        return {
            "agents": agent_status,
            "orchestrator_status": "healthy",
            "total_agents": len(agent_status)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agents status: {str(e)}")

# ============================================================================
# ENHANCED FILE IMPORTS
# ============================================================================

from fastapi.responses import FileResponse
from pathlib import Path

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Strategy AI Backend v2.0 is running!",
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    db_status = await db_manager.test_connection()
    pinecone_status = await pinecone_manager.test_connection()
    
    return {
        "status": "healthy" if db_status and pinecone_status else "degraded",
        "services": {
            "database": "connected" if db_status else "error",
            "vector_store": "connected" if pinecone_status else "error",
            "document_processor": "ready",
            "web_scraper": "ready",
            "ai_service": "ready"
        },
        "metrics": {
            "total_documents": await db_manager.get_document_count(),
            "total_sectors": await db_manager.get_sector_count(),
            "total_use_cases": await db_manager.get_use_case_count()
        }
    }

# ============================================================================
# CHAT & QUERY ENDPOINTS (Main User Interface)
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """
    Main chat endpoint - V0 frontend sends messages here
    Includes intelligent use case detection and custom prompt templates
    """
    try:
        # Auto-detect use case if not specified
        if not message.use_case:
            message.use_case = await ai_service.detect_use_case(
                message.message, message.sector
            )

        # Get custom prompt template if available
        prompt_template = await db_manager.get_prompt_template(
            message.sector, message.use_case
        )

        # Retrieve relevant documents from vector store
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

        # Format sources for V0 frontend
        formatted_sources = [
            {
                "document_title": doc.get("metadata", {}).get("title", "Untitled"),
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "relevance_score": doc.get("score", 0.0),
                "chunk_preview": doc.get("text", "")[:200] + "..."
            }
            for doc in relevant_docs[:5]
        ]

        # Log interaction for analytics
        await db_manager.log_chat_interaction(
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
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

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
# DOCUMENT MANAGEMENT ENDPOINTS (Admin Features)
# ============================================================================

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    sector: str = Form("General"),
    use_case: Optional[str] = Form(None),
    tags: Optional[str] = Form(""),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and process documents with AI metadata generation"""
    try:
        # Validate file type
        allowed_types = {".pdf", ".docx", ".txt", ".csv", ".md"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}"
            )

        doc_id = str(uuid.uuid4())

        # Save file temporarily
        temp_file_path = f"/tmp/{doc_id}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Process in background (like your Streamlit processing)
        background_tasks.add_task(
            process_uploaded_document,
            doc_id=doc_id,
            file_path=temp_file_path,
            original_filename=file.filename,
            title=title,
            sector=sector,
            use_case=use_case,
            tags=tags,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "message": "Document uploaded successfully. Processing in background...",
            "status_endpoint": f"/api/documents/{doc_id}/status"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/api/documents/scrape-url")
async def scrape_url(background_tasks: BackgroundTasks, request: URLScrapeRequest):
    """Comprehensive web scraping with navigation support"""
    try:
        job_id = str(uuid.uuid4())

        # Start scraping in background
        background_tasks.add_task(
            process_web_scraping,
            job_id=job_id,
            url=request.url,
            depth=request.depth,
            max_pages=request.max_pages,
            follow_nav_links=request.follow_nav_links,
            include_downloads=request.include_downloads,
            sector=request.sector,
            use_case=request.use_case
        )

        return {
            "success": True,
            "job_id": job_id,
            "url": request.url,
            "message": "Web scraping started. Processing in background...",
            "status_endpoint": f"/api/documents/scraping/{job_id}/status"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting web scraping: {str(e)}")

# ============================================================================
# CSV IMPORT/EXPORT ENDPOINTS (Key Admin Features)
# ============================================================================

@app.get("/api/admin/documents/export-csv")
async def export_documents_csv(admin_key: str = Depends(admin_auth.verify_admin)):
    """Export all document metadata as CSV for Excel editing"""
    try:
        documents = await db_manager.get_all_documents_for_export()

        # Create CSV response (like your Streamlit export)
        output = io.StringIO()
        if documents:
            writer = csv.DictWriter(output, fieldnames=documents[0].keys())
            writer.writeheader()
            writer.writerows(documents)

        csv_content = output.getvalue()
        output.close()

        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=strategy_documents.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

@app.post("/api/admin/documents/import-csv")
async def import_documents_csv(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    update_existing: bool = Form(True),
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Import edited CSV to update document metadata"""
    try:
        # Read and validate CSV (like your Streamlit import)
        csv_content = await csv_file.read()
        csv_string = csv_content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        csv_data = list(csv_reader)

        if not csv_data:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Validate required columns
        required_columns = ["document_id", "title", "sector"]
        missing_columns = [col for col in required_columns if col not in csv_data[0].keys()]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )

        import_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_csv_import,
            import_id=import_id,
            csv_data=csv_data,
            update_existing=update_existing
        )

        return {
            "success": True,
            "import_id": import_id,
            "records_to_process": len(csv_data),
            "message": "CSV import started. Processing in background...",
            "status_endpoint": f"/api/admin/imports/{import_id}/status"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing CSV: {str(e)}")

# ============================================================================
# SECTOR & USE CASE MANAGEMENT (Dynamic Dropdown Management)
# ============================================================================

@app.get("/api/sectors")
async def list_sectors():
    """Get all sectors for V0 dropdown population"""
    try:
        sectors = await db_manager.get_all_sectors()
        return {"sectors": sectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sectors: {str(e)}")

@app.post("/api/sectors")
async def create_sector(sector: SectorCreate, admin_key: str = Depends(admin_auth.verify_admin)):
    """Create new sector (admin function)"""
    try:
        sector_id = await db_manager.create_sector(sector.name, sector.description)
        return {
            "success": True,
            "sector_id": sector_id,
            "name": sector.name,
            "message": "Sector created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating sector: {str(e)}")

@app.get("/api/use-cases")
async def list_use_cases(sector: Optional[str] = None):
    """Get use cases, filtered by sector for cascading dropdowns"""
    try:
        use_cases = await db_manager.get_use_cases(sector=sector)
        return {"use_cases": use_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing use cases: {str(e)}")

@app.post("/api/use-cases")
async def create_use_case(use_case: UseCaseCreate, admin_key: str = Depends(admin_auth.verify_admin)):
    """Create use case with custom prompt template"""
    try:
        use_case_id = await db_manager.create_use_case(
            name=use_case.name,
            sector=use_case.sector,
            tags=use_case.tags,
            prompt_template=use_case.prompt_template
        )
        
        return {
            "success": True,
            "use_case_id": use_case_id,
            "name": use_case.name,
            "sector": use_case.sector,
            "message": "Use case created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating use case: {str(e)}")

# ============================================================================
# DOCUMENT LISTING & MANAGEMENT (For V0 Admin Dashboard)
# ============================================================================

@app.get("/api/documents")
async def list_documents(
    sector: Optional[str] = None,
    use_case: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List documents with filtering for V0 admin interface"""
    try:
        documents = await db_manager.get_documents(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search,
            limit=limit,
            offset=offset
        )

        total_count = await db_manager.get_document_count(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search
        )

        return {
            "documents": documents,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.put("/api/documents/{document_id}/metadata")
async def update_document_metadata(document_id: str, updates: MetadataUpdate):
    """Update individual document metadata (like your Streamlit edit function)"""
    try:
        # Update in database
        success = await db_manager.update_document_metadata(document_id, updates.dict(exclude_unset=True))
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update in vector store
        await pinecone_manager.update_document_metadata(document_id, updates.dict(exclude_unset=True))

        return {
            "success": True,
            "document_id": document_id,
            "message": "Metadata updated successfully",
            "updated_fields": list(updates.dict(exclude_unset=True).keys())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating metadata: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, admin_key: str = Depends(admin_auth.verify_admin)):
    """Delete document and all chunks (like your Streamlit delete function)"""
    try:
        # Delete from vector store first
        await pinecone_manager.delete_document(document_id)

        # Delete from database
        success = await db_manager.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "success": True,
            "document_id": document_id,
            "message": "Document deleted successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# ============================================================================
# BACKGROUND PROCESSING FUNCTIONS (Like your Streamlit background tasks)
# ============================================================================

async def process_uploaded_document(
    doc_id: str, file_path: str, original_filename: str,
    title: Optional[str], sector: str, use_case: Optional[str],
    tags: Optional[str], chunk_size: int, chunk_overlap: int
):
    """Background processing for uploaded documents"""
    try:
        await db_manager.update_document_status(doc_id, "processing")

        # Process document (like your document_processor.py)
        chunks = await doc_processor.process_file(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Generate AI metadata for each chunk (like your auto_tagger.py)
        for i, chunk in enumerate(chunks):
            metadata = await ai_service.generate_chunk_metadata(
                text=chunk["text"],
                sector=sector,
                use_case=use_case,
                suggested_title=title or original_filename
            )
            chunk["metadata"] = metadata

        # Store in database (like your vector_db.py)
        await db_manager.create_document(
            document_id=doc_id, title=title or original_filename,
            filename=original_filename, sector=sector, use_case=use_case,
            tags=tags, chunks=chunks, source_type="file"
        )

        # Store in Pinecone
        await pinecone_manager.store_document_chunks(doc_id, chunks)

        await db_manager.update_document_status(doc_id, "completed")

        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        await db_manager.update_document_status(doc_id, "failed", error=str(e))
        if os.path.exists(file_path):
            os.remove(file_path)

async def process_web_scraping(
    job_id: str, url: str, depth: str, max_pages: int,
    follow_nav_links: bool, include_downloads: bool,
    sector: str, use_case: Optional[str]
):
    """Background web scraping processing"""
    try:
        await db_manager.update_scraping_status(job_id, "processing")

        # Scrape website (comprehensive like you wanted)
        scraped_content = await web_scraper.scrape_comprehensive(
            start_url=url, depth=depth, max_pages=max_pages,
            follow_nav_links=follow_nav_links, include_downloads=include_downloads
        )

        # Process each scraped page as separate document
        for page_data in scraped_content:
            page_doc_id = f"{job_id}_page_{hash(page_data['url'])}"

            chunks = await doc_processor.process_text(
                text=page_data["content"],
                source_url=page_data["url"]
            )

            # Generate metadata for each chunk
            for chunk in chunks:
                metadata = await ai_service.generate_chunk_metadata(
                    text=chunk["text"], sector=sector, use_case=use_case,
                    suggested_title=page_data["title"], source_url=page_data["url"]
                )
                chunk["metadata"] = metadata

            # Store in database
            await db_manager.create_document(
                document_id=page_doc_id, title=page_data["title"],
                filename=None, sector=sector, use_case=use_case,
                tags="web_scraped", chunks=chunks, source_type="url",
                source_url=page_data["url"]
            )

            # Store in Pinecone
            await pinecone_manager.store_document_chunks(page_doc_id, chunks)

        await db_manager.update_scraping_status(job_id, "completed")

    except Exception as e:
        await db_manager.update_scraping_status(job_id, "failed", error=str(e))

async def process_csv_import(import_id: str, csv_data: List[Dict], update_existing: bool):
    """Background CSV import processing (like your CSV import feature)"""
    try:
        await db_manager.update_import_status(import_id, "processing")

        updated_count = 0
        created_count = 0
        failed_count = 0

        for row in csv_data:
            try:
                doc_id = row.get("document_id")
                
                if update_existing and doc_id:
                    # Update existing
                    success = await db_manager.update_document_from_csv_row(row)
                    if success:
                        await pinecone_manager.update_document_metadata(doc_id, row)
                        updated_count += 1
                    else:
                        failed_count += 1
                else:
                    # Create new if content provided
                    if row.get("content") or row.get("chunk_text"):
                        await db_manager.create_document_from_csv_row(row)
                        created_count += 1
                    else:
                        failed_count += 1

            except Exception as e:
                failed_count += 1

        await db_manager.update_import_status(
            import_id, "completed",
            stats={"updated": updated_count, "created": created_count, "failed": failed_count}
        )

    except Exception as e:
        await db_manager.update_import_status(import_id, "failed", error=str(e))

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )