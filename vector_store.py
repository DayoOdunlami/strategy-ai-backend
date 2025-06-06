# vector_store.py - Vector Store Manager for Semantic Search
import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
from config import Settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Vector store manager for semantic search and document retrieval
    Handles embeddings, similarity search, and context retrieval
    """
    
    def __init__(self):
        self.settings = Settings()
        self.demo_mode = True  # Start in demo mode
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
                    "vector": [0.1, 0.2, 0.3],  # Mock embedding
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
                    "vector": [0.4, 0.5, 0.6],  # Mock embedding
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
                    "vector": [0.7, 0.8, 0.9],  # Mock embedding
                    "created_at": datetime.now()
                }
            },
            "indexes": {
                "transport": ["chunk_1", "chunk_3"],
                "energy": ["chunk_2"],
                "general": []
            }
        }

    async def test_connection(self) -> bool:
        """Test vector store connection"""
        try:
            # In real implementation, test Pinecone/other vector DB
            return True
        except Exception as e:
            logger.error(f"Vector store connection test failed: {e}")
            return False

    # ============================================================================
    # EMBEDDING & STORAGE
    # ============================================================================

    async def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store document chunks with embeddings"""
        try:
            stored_count = 0
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Generate mock embedding (in real implementation, use OpenAI embeddings)
                mock_vector = [float(j % 100) / 100 for j in range(100)]
                
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk.get("text", ""),
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "chunk_index": i,
                        **chunk.get("metadata", {})
                    },
                    "vector": mock_vector,
                    "created_at": datetime.now()
                }
                
                self.mock_vectors["chunks"][chunk_id] = chunk_data
                
                # Update sector index
                sector = metadata.get("sector", "general").lower()
                if sector not in self.mock_vectors["indexes"]:
                    self.mock_vectors["indexes"][sector] = []
                self.mock_vectors["indexes"][sector].append(chunk_id)
                
                stored_count += 1
            
            logger.info(f"Stored {stored_count} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return False

    async def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            chunks_to_delete = []
            
            for chunk_id, chunk_data in self.mock_vectors["chunks"].items():
                if chunk_data["metadata"].get("document_id") == document_id:
                    chunks_to_delete.append(chunk_id)
            
            for chunk_id in chunks_to_delete:
                del self.mock_vectors["chunks"][chunk_id]
                
                # Remove from indexes
                for sector, chunk_list in self.mock_vectors["indexes"].items():
                    if chunk_id in chunk_list:
                        chunk_list.remove(chunk_id)
            
            logger.info(f"Deleted {len(chunks_to_delete)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    # ============================================================================
    # SEMANTIC SEARCH
    # ============================================================================

    async def semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform semantic search for relevant chunks"""
        try:
            # In demo mode, return intelligent mock results based on query
            results = await self._mock_semantic_search(query, filters, top_k)
            
            logger.debug(f"Semantic search for '{query[:50]}...' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def _mock_semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """Mock semantic search with intelligent matching"""
        
        query_lower = query.lower()
        filtered_chunks = []
        
        # Apply filters
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
            
            # Calculate relevance score based on keyword matching
            text_lower = chunk_data["text"].lower()
            score = 0.0
            
            # Keyword matching
            if "transport" in query_lower and "transport" in text_lower:
                score += 0.9
            elif "energy" in query_lower and "energy" in text_lower:
                score += 0.9
            elif "strategy" in query_lower and "strategy" in text_lower:
                score += 0.8
            elif "project" in query_lower and "project" in text_lower:
                score += 0.7
            elif "framework" in query_lower and "framework" in text_lower:
                score += 0.7
            elif "lessons" in query_lower and "lessons" in text_lower:
                score += 0.8
            else:
                # General relevance
                score = 0.6
            
            # Boost score for exact sector match
            if filters and "sector" in filters:
                if metadata.get("sector", "").lower() == filters["sector"].lower():
                    score += 0.1
            
            if score > 0.0:
                filtered_chunks.append({
                    "id": chunk_id,
                    "text": chunk_data["text"],
                    "metadata": metadata,
                    "score": score
                })
        
        # Sort by relevance score
        filtered_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top results
        return filtered_chunks[:top_k]

    # ============================================================================
    # CONTEXT PREPARATION
    # ============================================================================

    async def get_context_for_query(
        self,
        query: str,
        sector: str = "General",
        use_case: Optional[str] = None,
        max_chunks: int = 8
    ) -> List[Dict[str, Any]]:
        """Get relevant context chunks for a query"""
        
        filters = {"sector": sector}
        if use_case:
            filters["use_case"] = use_case
        
        context_chunks = await self.semantic_search(
            query=query,
            filters=filters,
            top_k=max_chunks
        )
        
        return context_chunks

    async def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents similar to the given document"""
        
        # Get chunks from the source document
        source_chunks = []
        for chunk_id, chunk_data in self.mock_vectors["chunks"].items():
            if chunk_data["metadata"].get("document_id") == document_id:
                source_chunks.append(chunk_data)
        
        if not source_chunks:
            return []
        
        # In real implementation, use embeddings to find similar documents
        # For demo, return documents from same sector
        source_sector = source_chunks[0]["metadata"].get("sector", "General")
        
        similar_docs = {}
        for chunk_id, chunk_data in self.mock_vectors["chunks"].items():
            chunk_doc_id = chunk_data["metadata"].get("document_id")
            chunk_sector = chunk_data["metadata"].get("sector", "General")
            
            if chunk_doc_id != document_id and chunk_sector == source_sector:
                if chunk_doc_id not in similar_docs:
                    similar_docs[chunk_doc_id] = {
                        "document_id": chunk_doc_id,
                        "title": chunk_data["metadata"].get("title", "Unknown"),
                        "sector": chunk_sector,
                        "similarity_score": 0.8,  # Mock score
                        "matching_chunks": 0
                    }
                similar_docs[chunk_doc_id]["matching_chunks"] += 1
        
        # Sort by matching chunks (proxy for similarity)
        results = list(similar_docs.values())
        results.sort(key=lambda x: x["matching_chunks"], reverse=True)
        
        return results[:top_k]

    # ============================================================================
    # ANALYTICS
    # ============================================================================

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
            "indexes": {k: len(v) for k, v in self.mock_vectors["indexes"].items()},
            "avg_chunk_length": sum(
                len(chunk["text"]) for chunk in self.mock_vectors["chunks"].values()
            ) / total_chunks if total_chunks > 0 else 0
        }

# Global vector store manager instance
vector_store = VectorStoreManager() 