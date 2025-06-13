# database.py - Database Manager for Multi-Agent Strategy AI (Demo Mode)
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for multi-agent Strategy AI system
    Currently in demo mode with mock data
    """
    
    def __init__(self):
        self.demo_mode = True
        self.mock_data = self._initialize_mock_data()
        logger.info("Database manager initialized in demo mode")

    def _initialize_mock_data(self) -> Dict[str, Any]:
        """Initialize mock data for demo mode"""
        return {
            "documents": {},
            "feedback": {},
            "sectors": {
                "transport": {"name": "Transport", "description": "Transport sector strategies", "created_at": datetime.now()},
                "energy": {"name": "Energy", "description": "Energy sector strategies", "created_at": datetime.now()},
                "general": {"name": "General", "description": "General strategic guidance", "created_at": datetime.now()}
            },
            "use_cases": {
                "quick_playbook": {"name": "Quick Playbook Answers", "sector": "General", "created_at": datetime.now()},
                "lessons_learned": {"name": "Lessons Learned", "sector": "General", "created_at": datetime.now()},
                "project_review": {"name": "Project Review / MOT", "sector": "General", "created_at": datetime.now()}
            },
            "chat_logs": {},
            "prompt_templates": {}
        }

    async def test_connection(self) -> bool:
        """Test database connection"""
        return True

    async def store_document(
        self,
        title: str,
        filename: str,
        sector: str,
        use_case: Optional[str] = None,
        source_type: str = "file",
        source_url: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store document metadata"""
        doc_id = str(uuid.uuid4())
        document = {
            "id": doc_id,
            "title": title,
            "filename": filename,
            "sector": sector,
            "use_case": use_case,
            "source_type": source_type,
            "source_url": source_url,
            "status": "completed",  # Demo mode - always completed
            "chunk_count": 5,  # Mock chunk count
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": metadata or {},
            "feedback_count": 0,
            "average_rating": None
        }
        
        self.mock_data["documents"][doc_id] = document
        logger.info(f"Stored document: {title} (ID: {doc_id})")
        return doc_id

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        return self.mock_data["documents"].get(document_id)

    async def list_documents(
        self,
        sector: Optional[str] = None,
        use_case: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        min_rating: Optional[float] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List documents with filtering"""
        docs = list(self.mock_data["documents"].values())
        
        # Apply filters
        if sector:
            docs = [d for d in docs if d.get("sector", "").lower() == sector.lower()]
        if use_case:
            docs = [d for d in docs if d.get("use_case", "").lower() == use_case.lower()]
        if source_type:
            docs = [d for d in docs if d.get("source_type", "").lower() == source_type.lower()]
        if search:
            search_lower = search.lower()
            docs = [d for d in docs if search_lower in d.get("title", "").lower()]
        if min_rating:
            docs = [d for d in docs if d.get("average_rating", 0) >= min_rating]
        
        total = len(docs)
        docs = docs[offset:offset + limit]
        
        return docs, total

    async def update_document_status(self, document_id: str, status: str, chunk_count: int = 0) -> bool:
        """Update document processing status"""
        if document_id in self.mock_data["documents"]:
            self.mock_data["documents"][document_id]["status"] = status
            self.mock_data["documents"][document_id]["chunk_count"] = chunk_count
            self.mock_data["documents"][document_id]["updated_at"] = datetime.now()
            return True
        return False

    async def delete_document(self, document_id: str) -> bool:
        """Delete document"""
        if document_id in self.mock_data["documents"]:
            del self.mock_data["documents"][document_id]
            return True
        return False

    async def store_feedback(
        self,
        chat_log_id: Optional[str] = None,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None,
        rating: Optional[int] = None,
        feedback_type: str = "general",
        comment: Optional[str] = None,
        helpful: Optional[bool] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store user feedback"""
        feedback_id = str(uuid.uuid4())
        feedback = {
            "id": feedback_id,
            "chat_log_id": chat_log_id,
            "document_id": document_id,
            "session_id": session_id,
            "rating": rating,
            "feedback_type": feedback_type,
            "comment": comment,
            "helpful": helpful,
            "created_at": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.mock_data["feedback"][feedback_id] = feedback
        
        # Update document rating if applicable
        if document_id and rating:
            await self._update_document_rating(document_id, rating)
        
        logger.info(f"Stored feedback: {feedback_type} (ID: {feedback_id})")
        return feedback_id

    async def _update_document_rating(self, document_id: str, rating: int):
        """Update document average rating"""
        if document_id in self.mock_data["documents"]:
            doc = self.mock_data["documents"][document_id]
            
            # Calculate new average (simplified)
            current_rating = doc.get("average_rating", 0)
            current_count = doc.get("feedback_count", 0)
            
            new_count = current_count + 1
            new_average = ((current_rating * current_count) + rating) / new_count
            
            doc["average_rating"] = round(new_average, 2)
            doc["feedback_count"] = new_count

    async def get_feedback_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback analytics"""
        feedback_list = list(self.mock_data["feedback"].values())
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_feedback = [f for f in feedback_list if f["created_at"] >= cutoff_date]
        
        total_feedback = len(recent_feedback)
        ratings = [f["rating"] for f in recent_feedback if f["rating"]]
        helpful_votes = [f["helpful"] for f in recent_feedback if f["helpful"] is not None]
        
        return {
            "total_feedback": total_feedback,
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "helpful_percentage": sum(helpful_votes) / len(helpful_votes) * 100 if helpful_votes else 0,
            "recent_feedback": recent_feedback[-10:],  # Last 10
            "rating_distribution": {str(i): ratings.count(i) for i in range(1, 6)},
            "feedback_by_type": {},
            "feedback_trends": {}
        }

    async def log_chat_interaction(
        self,
        message: str,
        response: str,
        sector: str,
        use_case: Optional[str],
        session_id: Optional[str],
        user_type: str,
        confidence: float,
        sources: List[Dict[str, Any]],
        agents_used: Optional[List[str]] = None,
        model_used: Optional[str] = None
    ) -> str:
        """Log chat interaction"""
        chat_id = str(uuid.uuid4())
        chat_log = {
            "id": chat_id,
            "message": message,
            "response": response,
            "sector": sector,
            "use_case": use_case,
            "session_id": session_id,
            "user_type": user_type,
            "confidence": confidence,
            "sources": sources,
            "agents_used": agents_used or [],
            "model_used": model_used,
            "timestamp": datetime.now()
        }
        
        self.mock_data["chat_logs"][chat_id] = chat_log
        return chat_id

    async def get_chat_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get chat history"""
        logs = list(self.mock_data["chat_logs"].values())
        
        if session_id:
            logs = [log for log in logs if log.get("session_id") == session_id]
        
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return logs[:limit]

    async def get_sectors(self) -> List[Dict[str, Any]]:
        """Get all sectors"""
        return list(self.mock_data["sectors"].values())

    async def get_use_cases(self, sector: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get use cases, optionally filtered by sector"""
        use_cases = list(self.mock_data["use_cases"].values())
        if sector:
            use_cases = [uc for uc in use_cases if uc.get("sector", "").lower() == sector.lower()]
        return use_cases

    async def get_prompt_template(self, sector: str, use_case: str) -> Optional[str]:
        """Get custom prompt template"""
        template_key = f"{sector}_{use_case}".lower()
        return self.mock_data["prompt_templates"].get(template_key)

    async def get_document_count(self) -> int:
        """Get total document count"""
        return len(self.mock_data["documents"])

    async def get_sector_count(self) -> int:
        """Get total sector count"""
        return len(self.mock_data["sectors"])

    async def get_use_case_count(self) -> int:
        """Get total use case count"""
        return len(self.mock_data["use_cases"])

    async def get_feedback_count(self) -> int:
        """Get total feedback count"""
        return len(self.mock_data["feedback"])

    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        docs = list(self.mock_data["documents"].values())
        
        # Documents by sector
        docs_by_sector = {}
        for doc in docs:
            sector = doc.get("sector", "Unknown")
            docs_by_sector[sector] = docs_by_sector.get(sector, 0) + 1

        # Documents by use case
        docs_by_use_case = {}
        for doc in docs:
            use_case = doc.get("use_case", "Unknown")
            docs_by_use_case[use_case] = docs_by_use_case.get(use_case, 0) + 1

        # Documents by source type
        docs_by_source = {}
        for doc in docs:
            source_type = doc.get("source_type", "Unknown")
            docs_by_source[source_type] = docs_by_source.get(source_type, 0) + 1

        return {
            "total_documents": len(docs),
            "total_chunks": sum(doc.get("chunk_count", 0) for doc in docs),
            "total_sectors": len(self.mock_data["sectors"]),
            "total_use_cases": len(self.mock_data["use_cases"]),
            "documents_by_sector": docs_by_sector,
            "documents_by_use_case": docs_by_use_case,
            "documents_by_source_type": docs_by_source,
            "recent_activity_count": len([d for d in docs if (datetime.now() - d["created_at"]).days <= 7]),
            "storage_usage": {"documents": len(docs), "feedback": len(self.mock_data["feedback"])}
        }

    async def update_document_fields(self, document_id: str, update_data: dict) -> bool:
        try:
            update_data['updated_at'] = datetime.now().isoformat()
            self.supabase.table('documents').update(update_data).eq('id', document_id).execute()
            return True
        except Exception as e:
            logging.error(f"Failed to update document fields: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager() 