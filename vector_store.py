# vector_store.py - Vector Store Manager (Demo Mode)
import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Vector store manager for semantic search (demo mode)
    Provides intelligent mock search results
    """
    
    def __init__(self):
        self.demo_mode = True
        self.mock_vectors = self._initialize_mock_vectors()
        logger.info("Vector store manager initialized in demo mode")

    def _initialize_mock_vectors(self) -> Dict[str, Any]:
        """Initialize mock vector data for demo mode"""
        return {
            "chunks": {
                "chunk_1": {
                    "id": "chunk_1",
                    "text": "Transport strategy development requires comprehensive analysis of infrastructure, technology adoption, and user needs. Key considerations include modal integration, sustainability objectives, and digital transformation initiatives.",
                    "metadata": {
                        "document_id": "doc_transport_1",
                        "title": "Transport Strategy Framework 2024",
                        "sector": "Transport",
                        "use_case": "Quick Playbook Answers",
                        "source": "Connected Places Catapult",
                        "chunk_index": 0
                    },
                    "created_at": datetime.now()
                },
                "chunk_2": {
                    "id": "chunk_2", 
                    "text": "Energy transition strategies must balance renewable energy deployment with grid stability and energy security. Critical factors include storage integration, demand response mechanisms, and regulatory frameworks.",
                    "metadata": {
                        "document_id": "doc_energy_1",
                        "title": "Energy Transition Roadmap",
                        "sector": "Energy",
                        "use_case": "Quick Playbook Answers",
                        "source": "Energy Policy Institute",
                        "chunk_index": 0
                    },
                    "created_at": datetime.now()
                },
                "chunk_3": {
                    "id": "chunk_3",
                    "text": "Lessons learned from previous transport projects highlight the importance of stakeholder engagement, phased implementation approaches, and continuous monitoring of key performance indicators.",
                    "metadata": {
                        "document_id": "doc_transport_2",
                        "title": "Transport Project Lessons Learned",
                        "sector": "Transport", 
                        "use_case": "Lessons Learned",
                        "source": "Project Review Board",
                        "chunk_index": 0
                    },
                    "created_at": datetime.now()
                }
            }
        }

    async def test_connection(self) -> bool:
        """Test vector store connection"""
        return True

    async def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store document chunks (demo mode)"""
        logger.info(f"Demo: Stored {len(chunks)} chunks for document {document_id}")
        return True

    async def delete_document_chunks(self, document_id: str) -> bool:
        """Delete document chunks (demo mode)"""
        logger.info(f"Demo: Deleted chunks for document {document_id}")
        return True

    async def semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform semantic search for relevant chunks"""
        try:
            results = self._mock_semantic_search(query, filters, top_k)
            logger.debug(f"Semantic search for '{query[:50]}...' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _mock_semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """Mock semantic search with intelligent matching"""
        
        query_lower = query.lower()
        filtered_chunks = []
        
        # Apply filters and calculate relevance
        for chunk_id, chunk_data in self.mock_vectors["chunks"].items():
            metadata = chunk_data["metadata"]
            
            # Sector filter
            if filters and "sector" in filters:
                if metadata.get("sector", "").lower() != filters["sector"].lower():
                    continue
            
            # Use case filter  
            if filters and "use_case" in filters:
                if metadata.get("use_case", "").lower() != filters["use_case"].lower():
                    continue
            
            # Calculate relevance score
            text_lower = chunk_data["text"].lower()
            score = 0.6  # Base score
            
            # Keyword matching
            if "transport" in query_lower and "transport" in text_lower:
                score += 0.3
            if "energy" in query_lower and "energy" in text_lower:
                score += 0.3
            if "strategy" in query_lower and "strategy" in text_lower:
                score += 0.2
            if "lessons" in query_lower and "lessons" in text_lower:
                score += 0.2
            
            # Boost for sector match
            if filters and "sector" in filters:
                if metadata.get("sector", "").lower() == filters["sector"].lower():
                    score += 0.1
            
            filtered_chunks.append({
                "id": chunk_id,
                "text": chunk_data["text"],
                "metadata": metadata,
                "score": score
            })
        
        # Sort by relevance
        filtered_chunks.sort(key=lambda x: x["score"], reverse=True)
        return filtered_chunks[:top_k]

    async def get_vector_analytics(self) -> Dict[str, Any]:
        """Get vector store analytics"""
        total_chunks = len(self.mock_vectors["chunks"])
        
        chunks_by_sector = {}
        chunks_by_use_case = {}
        
        for chunk_data in self.mock_vectors["chunks"].values():
            metadata = chunk_data["metadata"]
            
            sector = metadata.get("sector", "Unknown")
            chunks_by_sector[sector] = chunks_by_sector.get(sector, 0) + 1
            
            use_case = metadata.get("use_case", "Unknown")
            chunks_by_use_case[use_case] = chunks_by_use_case.get(use_case, 0) + 1
        
        return {
            "total_chunks": total_chunks,
            "total_documents": len(set(
                chunk["metadata"].get("document_id") 
                for chunk in self.mock_vectors["chunks"].values()
            )),
            "chunks_by_sector": chunks_by_sector,
            "chunks_by_use_case": chunks_by_use_case,
            "avg_chunk_length": sum(
                len(chunk["text"]) for chunk in self.mock_vectors["chunks"].values()
            ) / total_chunks if total_chunks > 0 else 0
        }

# Global vector store manager instance
vector_store = VectorStoreManager() 