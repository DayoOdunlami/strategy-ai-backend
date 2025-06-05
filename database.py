# database.py - Supabase Database Manager (WITH FEEDBACK SYSTEM)
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import json
import logging
import uuid
from config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages all database operations using Supabase
    Enhanced with user feedback system
    """
    
    def __init__(self):
        self.supabase_url = settings.SUPABASE_URL
        self.supabase_key = settings.SUPABASE_SERVICE_KEY
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            result = self.client.table("documents").select("count").execute()
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    # ============================================================================
    # USER FEEDBACK METHODS (NEW)
    # ============================================================================

    async def store_feedback(self, feedback_data: Dict[str, Any]):
        """Store user feedback"""
        try:
            self.client.table("user_feedback").insert(feedback_data).execute()
            logger.info(f"Stored feedback: {feedback_data['id']}")
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            raise

    async def get_feedback_analytics(self, days: int = 30, feedback_type: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        try:
            # Base query for recent feedback
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = self.client.table("user_feedback").select("*").gte("created_at", cutoff_date)
            
            if feedback_type:
                query = query.eq("feedback_type", feedback_type)
                
            feedback_result = query.execute()
            feedback_data = feedback_result.data
            
            if not feedback_data:
                return {
                    "total_feedback": 0,
                    "average_rating": 0.0,
                    "helpful_percentage": 0.0,
                    "rating_distribution": {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0},
                    "feedback_by_type": {},
                    "feedback_trends": {},
                    "recent_feedback": []
                }
            
            # Calculate analytics
            ratings = [f["rating"] for f in feedback_data if f["rating"]]
            helpful_votes = [f["helpful"] for f in feedback_data if f["helpful"] is not None]
            
            # Rating distribution
            rating_distribution = {str(i): 0 for i in range(1, 6)}
            for rating in ratings:
                if rating:
                    rating_distribution[str(rating)] = rating_distribution.get(str(rating), 0) + 1
            
            # Feedback by type
            feedback_by_type = {}
            for feedback in feedback_data:
                ftype = feedback.get("feedback_type", "general")
                feedback_by_type[ftype] = feedback_by_type.get(ftype, 0) + 1
            
            # Trends (last 7 days)
            feedback_trends = {}
            for i in range(7):
                date = (datetime.utcnow() - timedelta(days=i)).date()
                date_str = date.isoformat()
                daily_feedback = [f for f in feedback_data if f["created_at"].startswith(date_str)]
                feedback_trends[date_str] = len(daily_feedback)
            
            analytics = {
                "total_feedback": len(feedback_data),
                "average_rating": sum(ratings) / len(ratings) if ratings else 0.0,
                "helpful_percentage": (sum(helpful_votes) / len(helpful_votes) * 100) if helpful_votes else 0.0,
                "rating_distribution": rating_distribution,
                "feedback_by_type": feedback_by_type,
                "feedback_trends": feedback_trends,
                "recent_feedback": sorted(feedback_data, key=lambda x: x["created_at"], reverse=True)[:10]
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return {"error": str(e)}

    async def get_recent_feedback(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent feedback entries"""
        try:
            result = self.client.table("user_feedback").select("""
                id, rating, feedback_type, comment, helpful, created_at,
                chat_logs(query, sector, use_case),
                documents(title, sector)
            """).order("created_at", desc=True).limit(limit).execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []

    async def get_feedback_count(self) -> int:
        """Get total feedback count"""
        try:
            result = self.client.table("user_feedback").select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting feedback count: {e}")
            return 0

    async def get_document_feedback_summary(self, document_id: str) -> Dict[str, Any]:
        """Get feedback summary for a specific document"""
        try:
            result = self.client.table("user_feedback").select("*").eq("document_id", document_id).execute()
            feedback_data = result.data
            
            if not feedback_data:
                return {"feedback_count": 0, "average_rating": None}
            
            ratings = [f["rating"] for f in feedback_data if f["rating"]]
            helpful_votes = [f["helpful"] for f in feedback_data if f["helpful"] is not None]
            
            return {
                "feedback_count": len(feedback_data),
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "helpful_percentage": (sum(helpful_votes) / len(helpful_votes) * 100) if helpful_votes else None,
                "total_comments": len([f for f in feedback_data if f.get("comment")])
            }
            
        except Exception as e:
            logger.error(f"Error getting document feedback summary: {e}")
            return {"error": str(e)}

    # ============================================================================
    # ENHANCED DOCUMENT MANAGEMENT (With Feedback Integration)
    # ============================================================================

    async def get_documents(
        self,
        sector: Optional[str] = None,
        use_case: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        min_rating: Optional[float] = None,  # New parameter
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get documents with filtering (enhanced with feedback data)"""
        try:
            # Build query with feedback data
            query = self.client.table("documents").select("""
                id, title, filename, sector, use_case, tags, source_type, source_url,
                status, chunk_count, created_at, updated_at,
                user_feedback(rating, helpful)
            """)
            
            # Apply filters
            if sector:
                query = query.eq("sector", sector)
            if use_case:
                query = query.eq("use_case", use_case)
            if source_type:
                query = query.eq("source_type", source_type)
            if search:
                query = query.ilike("title", f"%{search}%")
                
            query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
            
            result = query.execute()
            documents = result.data
            
            # Calculate feedback metrics for each document
            enhanced_documents = []
            for doc in documents:
                feedback_entries = doc.get("user_feedback", [])
                
                if feedback_entries:
                    ratings = [f["rating"] for f in feedback_entries if f["rating"]]
                    helpful_votes = [f["helpful"] for f in feedback_entries if f["helpful"] is not None]
                    
                    doc["feedback_count"] = len(feedback_entries)
                    doc["average_rating"] = sum(ratings) / len(ratings) if ratings else None
                    doc["helpful_percentage"] = (sum(helpful_votes) / len(helpful_votes) * 100) if helpful_votes else None
                else:
                    doc["feedback_count"] = 0
                    doc["average_rating"] = None
                    doc["helpful_percentage"] = None
                
                # Apply minimum rating filter if specified
                if min_rating is None or (doc["average_rating"] and doc["average_rating"] >= min_rating):
                    enhanced_documents.append(doc)
                
                # Remove the raw feedback data from response
                doc.pop("user_feedback", None)
            
            return enhanced_documents

        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []

    async def get_document_count(self, **filters) -> int:
        """Get total document count with optional filters (enhanced with rating filter)"""
        try:
            query = self.client.table("documents").select("id", count="exact")
            
            # Apply basic filters
            for key, value in filters.items():
                if key != "min_rating" and value:
                    query = query.eq(key, value)
                    
            result = query.execute()
            
            # If min_rating filter is specified, we need to get all documents and filter by rating
            if filters.get("min_rating"):
                # This is less efficient but necessary for rating filtering
                # In production, consider using a materialized view or cached ratings
                all_docs = await self.get_documents(**filters)
                return len(all_docs)
            
            return result.count or 0

        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    # ============================================================================
    # ENHANCED CHAT LOGGING (Returns Chat Log ID for Feedback)
    # ============================================================================

    async def log_chat_interaction(
        self,
        session_id: Optional[str],
        query: str,
        response: str,
        sources_used: List[str],
        sector: str,
        use_case: Optional[str]
    ) -> str:
        """Log chat interaction and return chat_log_id for feedback linking"""
        try:
            chat_log_id = str(uuid.uuid4())
            
            log_data = {
                "id": chat_log_id,
                "session_id": session_id,
                "query": query,
                "response": response[:1000],  # Truncate for storage
                "sources_used": json.dumps(sources_used),
                "sector": sector,
                "use_case": use_case,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.client.table("chat_logs").insert(log_data).execute()
            logger.info(f"Logged chat interaction: {chat_log_id}")
            
            return chat_log_id

        except Exception as e:
            logger.error(f"Error logging chat interaction: {e}")
            return ""

    # ============================================================================
    # ORIGINAL DOCUMENT MANAGEMENT METHODS (Preserved)
    # ============================================================================

    async def create_document(
        self, 
        document_id: str, 
        title: str, 
        filename: Optional[str],
        sector: str,
        use_case: Optional[str],
        tags: Optional[str],
        chunks: List[Dict],
        source_type: str,
        source_url: Optional[str] = None
    ):
        """Create a new document with chunks - like your Streamlit document processing"""
        try:
            # Insert main document record
            doc_data = {
                "id": document_id,
                "title": title,
                "filename": filename,
                "sector": sector,
                "use_case": use_case,
                "tags": tags,
                "source_type": source_type,
                "source_url": source_url,
                "status": "processing",
                "chunk_count": len(chunks),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.client.table("documents").insert(doc_data).execute()
            logger.info(f"Created document {document_id} with {len(chunks)} chunks")

            # Insert chunks (like your chunk storage)
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append({
                    "id": f"{document_id}_chunk_{i}",
                    "document_id": document_id,
                    "text": chunk["text"],
                    "chunk_index": i,
                    "metadata": chunk.get("metadata", {}),
                    "pinecone_id": f"doc_{document_id}_chunk_{i}",
                    "created_at": datetime.utcnow().isoformat()
                })

            if chunk_data:
                self.client.table("chunks").insert(chunk_data).execute()
                logger.info(f"Inserted {len(chunk_data)} chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Error creating document {document_id}: {e}")
            raise

    async def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """Get specific document with details (enhanced with feedback)"""
        try:
            result = self.client.table("documents").select("*").eq("id", document_id).execute()
            
            if result.data:
                document = result.data[0]
                
                # Add feedback summary
                feedback_summary = await self.get_document_feedback_summary(document_id)
                document.update(feedback_summary)
                
                return document
                
            return None
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None

    async def update_document_metadata(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update document metadata - like your Streamlit metadata editing"""
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            result = self.client.table("documents").update(updates).eq("id", document_id).execute()
            
            if result.data:
                logger.info(f"Updated metadata for document {document_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks - like your Streamlit delete function"""
        try:
            # Delete chunks first (foreign key constraint)
            self.client.table("chunks").delete().eq("document_id", document_id).execute()
            
            # Delete feedback related to this document
            self.client.table("user_feedback").delete().eq("document_id", document_id).execute()
            
            # Delete main document
            result = self.client.table("documents").delete().eq("id", document_id).execute()
            
            if result.data:
                logger.info(f"Deleted document {document_id} and its chunks/feedback")
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def update_document_status(self, document_id: str, status: str, error: Optional[str] = None):
        """Update document processing status"""
        try:
            updates = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            if error:
                updates["error_message"] = error
                
            self.client.table("documents").update(updates).eq("id", document_id).execute()
            logger.info(f"Updated status for document {document_id} to {status}")

        except Exception as e:
            logger.error(f"Error updating status for document {document_id}: {e}")

    # ============================================================================
    # SECTOR & USE CASE MANAGEMENT (Preserved)
    # ============================================================================

    async def get_all_sectors(self) -> List[Dict]:
        """Get all sectors for dropdown population"""
        try:
            result = self.client.table("sectors").select("*").order("name").execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting sectors: {e}")
            return []

    async def create_sector(self, name: str, description: str) -> str:
        """Create new sector"""
        try:
            sector_data = {
                "name": name,
                "description": description,
                "created_at": datetime.utcnow().isoformat()
            }
            result = self.client.table("sectors").insert(sector_data).execute()
            
            if result.data:
                logger.info(f"Created sector: {name}")
                return result.data[0]["id"]
            raise Exception("Failed to create sector")

        except Exception as e:
            logger.error(f"Error creating sector {name}: {e}")
            raise

    async def get_sector_count(self) -> int:
        """Get total sector count"""
        try:
            result = self.client.table("sectors").select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting sector count: {e}")
            return 0

    async def get_use_cases(self, sector: Optional[str] = None) -> List[Dict]:
        """Get use cases, optionally filtered by sector - for cascading dropdowns"""
        try:
            query = self.client.table("use_cases").select("*")
            
            if sector:
                query = query.eq("sector", sector)
                
            query = query.order("name")
            result = query.execute()
            return result.data

        except Exception as e:
            logger.error(f"Error getting use cases: {e}")
            return []

    async def create_use_case(
        self, 
        name: str, 
        sector: str, 
        tags: Optional[str],
        prompt_template: Optional[str]
    ) -> str:
        """Create new use case with custom prompt template"""
        try:
            use_case_data = {
                "name": name,
                "sector": sector,
                "tags": tags,
                "prompt_template": prompt_template,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            result = self.client.table("use_cases").insert(use_case_data).execute()
            
            if result.data:
                logger.info(f"Created use case: {name} for sector: {sector}")
                return result.data[0]["id"]
            raise Exception("Failed to create use case")

        except Exception as e:
            logger.error(f"Error creating use case {name}: {e}")
            raise

    async def get_prompt_template(self, sector: str, use_case: str) -> Optional[str]:
        """Get custom prompt template for sector/use case combination"""
        try:
            result = self.client.table("use_cases").select("prompt_template").eq("sector", sector).eq("name", use_case).execute()
            
            if result.data and result.data[0]["prompt_template"]:
                return result.data[0]["prompt_template"]
            return None

        except Exception as e:
            logger.error(f"Error getting prompt template for {sector}/{use_case}: {e}")
            return None

    async def get_use_case_info(self, sector: str, use_case: str) -> Dict[str, Any]:
        """Get detailed use case information"""
        try:
            result = self.client.table("use_cases").select("*").eq("sector", sector).eq("name", use_case).execute()
            
            if result.data:
                return result.data[0]
            return {}

        except Exception as e:
            logger.error(f"Error getting use case info for {sector}/{use_case}: {e}")
            return {}

    async def get_use_case_count(self) -> int:
        """Get total use case count"""
        try:
            result = self.client.table("use_cases").select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting use case count: {e}")
            return 0

    # ============================================================================
    # ENHANCED SYSTEM ANALYTICS (With Feedback)
    # ============================================================================

    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics including feedback data"""
        try:
            analytics = {}
            
            # Document counts by category
            total_docs = await self.get_document_count()
            analytics["total_documents"] = total_docs
            
            # Total chunks count
            chunks_result = self.client.table("chunks").select("id", count="exact").execute()
            analytics["total_chunks"] = chunks_result.count or 0
            
            # Sector and use case counts
            analytics["total_sectors"] = await self.get_sector_count()
            analytics["total_use_cases"] = await self.get_use_case_count()
            
            # Documents by sector
            sectors_result = self.client.table("documents").select("sector").execute()
            sector_counts = {}
            for doc in sectors_result.data:
                sector = doc.get("sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            analytics["documents_by_sector"] = sector_counts
            
            # Documents by use case
            use_cases_result = self.client.table("documents").select("use_case").execute()
            use_case_counts = {}
            for doc in use_cases_result.data:
                use_case = doc.get("use_case", "Unknown")
                use_case_counts[use_case] = use_case_counts.get(use_case, 0) + 1
            analytics["documents_by_use_case"] = use_case_counts
            
            # Documents by source type
            source_types_result = self.client.table("documents").select("source_type").execute()
            source_type_counts = {}
            for doc in source_types_result.data:
                source_type = doc.get("source_type", "Unknown")
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
            analytics["documents_by_source_type"] = source_type_counts
            
            # Recent activity (last 7 days)
            week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            recent_result = self.client.table("chat_logs").select("id", count="exact").gte(
                "timestamp", week_ago
            ).execute()
            analytics["recent_activity_count"] = recent_result.count or 0
            
            # Storage usage (approximate)
            analytics["storage_usage"] = {
                "documents": total_docs,
                "chunks": analytics["total_chunks"],
                "feedback_entries": await self.get_feedback_count()
            }
            
            return analytics

        except Exception as e:
            logger.error(f"Error getting system analytics: {e}")
            return {}

    # ============================================================================
    # CSV IMPORT/EXPORT & BACKGROUND TASKS (Preserved)
    # ============================================================================

    async def get_all_documents_for_export(self) -> List[Dict]:
        """Get all documents with metadata for CSV export - like your Streamlit export"""
        try:
            # Get documents with chunk information and feedback summary
            result = self.client.table("documents").select("""
                id, title, filename, sector, use_case, tags, source_type, source_url,
                status, chunk_count, created_at, updated_at
            """).execute()
            
            documents = result.data
            
            # Add feedback summary to each document
            for doc in documents:
                feedback_summary = await self.get_document_feedback_summary(doc["id"])
                doc.update(feedback_summary)
                
                # Also get chunk data for detailed export
                chunks_result = self.client.table("chunks").select("""
                    chunk_index, text, metadata
                """).eq("document_id", doc["id"]).order("chunk_index").execute()
                
                doc["chunks"] = chunks_result.data

            logger.info(f"Exported {len(documents)} documents for CSV")
            return documents

        except Exception as e:
            logger.error(f"Error exporting documents: {e}")
            return []

    async def update_scraping_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Update web scraping job status"""
        try:
            status_data = {
                "job_id": job_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if error:
                status_data["error"] = error
                
            # Upsert scraping status
            self.client.table("scraping_jobs").upsert(status_data).execute()
            logger.info(f"Updated scraping status for job {job_id}: {status}")

        except Exception as e:
            logger.error(f"Error updating scraping status for job {job_id}: {e}")

    async def update_import_status(
        self, 
        import_id: str, 
        status: str, 
        error: Optional[str] = None,
        stats: Optional[Dict[str, int]] = None
    ):
        """Update CSV import status"""
        try:
            status_data = {
                "import_id": import_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if error:
                status_data["error"] = error
            if stats:
                status_data["stats"] = json.dumps(stats)
                
            self.client.table("import_jobs").upsert(status_data).execute()
            logger.info(f"Updated import status for job {import_id}: {status}")

        except Exception as e:
            logger.error(f"Error updating import status for job {import_id}: {e}")

    async def get_recent_activity(self, limit: int = 20) -> List[Dict]:
        """Get recent system activity for admin monitoring"""
        try:
            result = self.client.table("chat_logs").select("""
                session_id, query, sector, use_case, timestamp
            """).order("timestamp", desc=True).limit(limit).execute()
            
            return result.data

        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []