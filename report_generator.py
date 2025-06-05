from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportCache:
    """Cache for report generation progress"""
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_access: Dict[str, datetime] = {}

    async def set_section(self, report_id: str, section: str, content: str):
        """Cache a report section"""
        if report_id not in self.cache:
            self.cache[report_id] = {
                "sections": {},
                "created_at": datetime.now(),
                "status": "in_progress"
            }
        self.cache[report_id]["sections"][section] = content
        self.last_access[report_id] = datetime.now()

    async def get_section(self, report_id: str, section: str) -> Optional[str]:
        """Get a cached section"""
        if report_id in self.cache:
            self.last_access[report_id] = datetime.now()
            return self.cache[report_id]["sections"].get(section)
        return None

    async def get_report_progress(self, report_id: str) -> dict:
        """Get report generation progress"""
        if report_id in self.cache:
            self.last_access[report_id] = datetime.now()
            sections = self.cache[report_id]["sections"]
            total_sections = len(self.report_templates.get(
                self.cache[report_id].get("report_type", ""),
                {"sections": []}
            )["sections"])
            
            return {
                "status": self.cache[report_id]["status"],
                "progress": len(sections) / total_sections if total_sections > 0 else 0,
                "completed_sections": list(sections.keys()),
                "created_at": self.cache[report_id]["created_at"].isoformat()
            }
        return None

    async def set_report_status(self, report_id: str, status: str):
        """Update report status"""
        if report_id in self.cache:
            self.cache[report_id]["status"] = status
            self.last_access[report_id] = datetime.now()

    async def cleanup_old_reports(self, max_age_hours: int = 24):
        """Clean up old cached reports"""
        current_time = datetime.now()
        expired_reports = [
            report_id
            for report_id, last_access in self.last_access.items()
            if current_time - last_access > timedelta(hours=max_age_hours)
        ]
        
        for report_id in expired_reports:
            self.cache.pop(report_id, None)
            self.last_access.pop(report_id, None)

class ReportGenerator:
    def __init__(self):
        self.orchestrator = OrchestrationAgent()
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.cache = ReportCache()
        
        # Report templates (existing code...)

    async def generate_report(
        self,
        report_type: str,
        parameters: Dict[str, Any],
        format: str = "pdf"
    ) -> Dict[str, Any]:
        """Main report generation function with progress caching"""
        try:
            report_id = str(uuid.uuid4())
            logger.info(f"Generating {report_type} report in {format} format")
            
            # Initialize cache
            self.cache.cache[report_id] = {
                "report_type": report_type,
                "created_at": datetime.now(),
                "status": "starting"
            }
            
            # Step 1: Gather data and context
            context_data = await self._gather_report_context(report_type, parameters)
            
            # Step 2: Generate report content using AI agents with caching
            report_content = await self._generate_report_content_with_cache(
                report_id,
                report_type,
                context_data,
                parameters
            )
            
            # Step 3: Create formatted documents
            await self.cache.set_report_status(report_id, "formatting")
            generated_files = []
            
            if format in ["pdf", "both"]:
                pdf_path = await self._generate_pdf_report(
                    report_id, report_type, report_content, parameters
                )
                if pdf_path:
                    generated_files.append({
                        "format": "pdf",
                        "path": pdf_path,
                        "filename": f"{report_type}_{report_id}.pdf"
                    })
            
            if format in ["docx", "both"]:
                docx_path = await self._generate_docx_report(
                    report_id, report_type, report_content, parameters
                )
                if docx_path:
                    generated_files.append({
                        "format": "docx",
                        "path": docx_path,
                        "filename": f"{report_type}_{report_id}.docx"
                    })
            
            # Step 4: Generate and save metadata
            await self.cache.set_report_status(report_id, "completing")
            report_metadata = {
                "report_id": report_id,
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "parameters": parameters,
                "files": generated_files,
                "word_count": len(report_content.get("full_content", "").split()),
                "sections_count": len(report_content.get("sections", {})),
                "ai_agents_used": report_content.get("agents_used", [])
            }
            
            # Save metadata
            await self._save_report_metadata(report_id, report_metadata)
            await self.cache.set_report_status(report_id, "completed")
            
            logger.info(f"Successfully generated report {report_id}")
            return {
                "success": True,
                "report_id": report_id,
                "metadata": report_metadata,
                "download_urls": [f"/api/reports/{report_id}/download/{file['filename']}" for file in generated_files]
            }
            
        except Exception as e:
            if report_id:
                await self.cache.set_report_status(report_id, "error")
            logger.error(f"Error generating report: {e}")
            return {
                "success": False,
                "error": str(e),
                "report_id": report_id if 'report_id' in locals() else None
            }

    async def _generate_report_content_with_cache(
        self,
        report_id: str,
        report_type: str,
        context_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate report content with section caching"""
        template = self.report_templates.get(report_type, {})
        sections = template.get("sections", [])
        
        # Prepare AI request
        ai_request = {
            "type": "report",
            "report_type": report_type,
            "context": json.dumps(context_data),
            "parameters": parameters,
            "sector": context_data.get("sector", "General")
        }
        
        generated_sections = {}
        for section in sections:
            # Check cache first
            cached_content = await self.cache.get_section(report_id, section)
            if cached_content:
                generated_sections[section] = cached_content
                continue
            
            # Generate new content
            section_content = await self.orchestrator.generate_section(
                section=section,
                report_type=report_type,
                parameters=parameters
            )
            
            # Cache the result
            await self.cache.set_section(report_id, section, section_content)
            generated_sections[section] = section_content
        
        return {
            "sections": generated_sections,
            "full_content": "\n\n".join(generated_sections.values()),
            "metadata": {
                "generated_by": "ReportAgent",
                "confidence": 0.9,
                "agents_used": ["ReportAgent", "AnalysisAgent"]
            },
            "template": template
        }

    async def cleanup_old_reports(self, max_age_hours: int = 24):
        """Clean up old report files and cache"""
        try:
            # Clean up cache
            await self.cache.cleanup_old_reports(max_age_hours)
            
            # Clean up files
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            for metadata_file in self.reports_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    generated_at = datetime.fromisoformat(metadata.get("generated_at", ""))
                    if generated_at < cutoff:
                        # Delete report files
                        for file_info in metadata.get("files", []):
                            file_path = Path(file_info["path"])
                            if file_path.exists():
                                file_path.unlink()
                        
                        # Delete metadata file
                        metadata_file.unlink()
                        
                except Exception as e:
                    logger.error(f"Error cleaning up report {metadata_file}: {e}")
            
        except Exception as e:
            logger.error(f"Error in cleanup_old_reports: {e}")

    async def get_report_progress(self, report_id: str) -> Optional[dict]:
        """Get report generation progress"""
        return await self.cache.get_report_progress(report_id)

# ... rest of the existing code ... 