# vector_store.py - Pinecone Vector Store Manager
import asyncio
from typing import List, Dict, Any, Optional
import logging
from pinecone import Pinecone, ServerlessSpec
import openai
import time
import json
from config import settings

logger = logging.getLogger(__name__)

class PineconeManager:
    """
    Pinecone vector store manager that handles all vector operations
    Replaces ChromaDB from your Streamlit version with production-ready Pinecone
    """
    
    def __init__(self):
        self.pinecone_api_key = settings.PINECONE_API_KEY
        self.openai_api_key = settings.OPENAI_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.environment = settings.PINECONE_ENVIRONMENT
        
        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("PINECONE_API_KEY and OPENAI_API_KEY must be set")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize OpenAI for embeddings
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Ensure index exists and get reference
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        logger.info(f"Initialized Pinecone manager with index: {self.index_name}")

    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-ada-002 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(1)
                    
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test Pinecone connection"""
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone connection successful. Index stats: {stats}")
            return True
        except Exception as e:
            logger.error(f"Pinecone connection test failed: {e}")
            return False

    # ============================================================================
    # EMBEDDING GENERATION
    # ============================================================================

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI - like your original embedding generation"""
        try:
            # Truncate text if too long (OpenAI limit is ~8191 tokens)
            if len(text) > 8000:
                text = text[:8000]
                logger.warning("Text truncated for embedding generation")
            
            response = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL_NAME,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        try:
            # OpenAI allows batch requests
            truncated_texts = []
            for text in texts:
                if len(text) > 8000:
                    truncated_texts.append(text[:8000])
                else:
                    truncated_texts.append(text)
            
            response = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL_NAME,
                input=truncated_texts,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    # ============================================================================
    # DOCUMENT STORAGE (Main Upload Function)
    # ============================================================================

    async def store_document_chunks(self, document_id: str, chunks: List[Dict]) -> bool:
        """
        Store document chunks in Pinecone with metadata
        This is the main function called after document processing
        """
        try:
            if not chunks:
                logger.warning(f"No chunks provided for document {document_id}")
                return False
            
            logger.info(f"Storing {len(chunks)} chunks for document {document_id}")
            
            # Extract texts for batch embedding generation
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings in batch for efficiency
            embeddings = await self.generate_batch_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create unique vector ID
                vector_id = f"doc_{document_id}_chunk_{i}"
                
                # Prepare metadata (Pinecone has metadata size limits)
                metadata = self._prepare_metadata(chunk, document_id, i)
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert vectors in batches (Pinecone limit: 100 vectors per request)
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch)
                    total_upserted += upsert_response.upserted_count
                    logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
                except Exception as e:
                    logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                    raise
            
            logger.info(f"Successfully stored {total_upserted} vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document chunks for {document_id}: {e}")
            raise

    def _prepare_metadata(self, chunk: Dict, document_id: str, chunk_index: int) -> Dict:
        """Prepare metadata for Pinecone (with size limitations)"""
        try:
            # Start with basic metadata
            metadata = {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "source_name": chunk.get("source_name", "unknown"),
                "token_count": chunk.get("token_count", 0)
            }
            
            # Add chunk metadata if available (from AI processing)
            chunk_metadata = chunk.get("metadata", {})
            
            # Add important fields with size limits
            for key in ["title", "topic", "tags", "sector", "use_case", "summary"]:
                if key in chunk_metadata:
                    value = str(chunk_metadata[key])
                    # Limit string length to avoid Pinecone metadata limits
                    metadata[key] = value[:500] if len(value) > 500 else value
            
            # Add text preview (truncated)
            text_preview = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            metadata["text_preview"] = text_preview
            
            # Add confidence score if available
            if "confidence" in chunk_metadata:
                metadata["confidence"] = float(chunk_metadata["confidence"])
            
            # Add source URL if available
            if chunk.get("source_url"):
                metadata["source_url"] = chunk["source_url"][:500]  # Limit URL length
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error preparing metadata: {e}")
            # Return minimal metadata on error
            return {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text_preview": chunk["text"][:100] + "..."
            }

    # ============================================================================
    # SEMANTIC SEARCH (Main Query Function)
    # ============================================================================

    async def semantic_search(
        self, 
        query: str, 
        filters: Optional[Dict] = None, 
        top_k: int = None
    ) -> List[Dict]:
        """
        Semantic search for relevant chunks - main function for RAG
        This replaces your Streamlit similarity_search function
        """
        try:
            top_k = top_k or settings.VECTOR_SEARCH_TOP_K
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Prepare Pinecone filter
            pinecone_filter = self._prepare_search_filter(filters)
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                filter=pinecone_filter,
                top_k=top_k,
                include_metadata=True,
                include_values=False  # We don't need the vectors back
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score),
                    "text": match.metadata.get("text_preview", ""),
                    "metadata": match.metadata
                }
                results.append(result)
            
            logger.info(f"Semantic search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _prepare_search_filter(self, filters: Optional[Dict]) -> Optional[Dict]:
        """Prepare filter for Pinecone search"""
        if not filters:
            return None
        
        pinecone_filter = {}
        
        # Convert filters to Pinecone format
        for key, value in filters.items():
            if value and value != "All":  # Skip empty or "All" filters
                if isinstance(value, list):
                    pinecone_filter[key] = {"$in": value}
                else:
                    pinecone_filter[key] = {"$eq": str(value)}
        
        return pinecone_filter if pinecone_filter else None

    # ============================================================================
    # METADATA UPDATES (Admin Functions)
    # ============================================================================

    async def update_document_metadata(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for all chunks of a document - like your Streamlit update function"""
        try:
            # First, find all vectors for this document
            query_result = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                filter={"document_id": {"$eq": document_id}},
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
                include_values=False
            )
            
            if not query_result.matches:
                logger.warning(f"No vectors found for document {document_id}")
                return False
            
            # Update each vector's metadata
            update_count = 0
            for match in query_result.matches:
                try:
                    # Merge existing metadata with updates
                    updated_metadata = {**match.metadata}
                    
                    # Apply updates with proper formatting
                    for key, value in updates.items():
                        if value is not None:  # Only update non-None values
                            # Ensure string values are truncated for Pinecone limits
                            if isinstance(value, str) and len(value) > 500:
                                updated_metadata[key] = value[:500]
                            else:
                                updated_metadata[key] = str(value)
                    
                    # Update the vector metadata
                    self.index.update(
                        id=match.id,
                        set_metadata=updated_metadata
                    )
                    update_count += 1
                    
                except Exception as e:
                    logger.error(f"Error updating vector {match.id}: {e}")
                    continue
            
            logger.info(f"Updated metadata for {update_count} vectors in document {document_id}")
            return update_count > 0
            
        except Exception as e:
            logger.error(f"Error updating document metadata for {document_id}: {e}")
            return False

    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document - like your Streamlit delete function"""
        try:
            # Delete by filter - Pinecone will delete all matching vectors
            delete_response = self.index.delete(
                filter={"document_id": {"$eq": document_id}}
            )
            
            logger.info(f"Deleted all vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    # ============================================================================
    # ANALYTICS & MONITORING
    # ============================================================================

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics for monitoring"""
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {},
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

    async def get_document_count(self) -> int:
        """Get total number of unique documents in index"""
        try:
            # Query for unique document IDs
            query_result = self.index.query(
                vector=[0.0] * 1536,
                filter={},
                top_k=10000,
                include_metadata=True,
                include_values=False
            )
            
            # Extract unique document IDs
            document_ids = set()
            for match in query_result.matches:
                doc_id = match.metadata.get("document_id")
                if doc_id:
                    document_ids.add(doc_id)
            
            return len(document_ids)
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    # ============================================================================
    # ADVANCED SEARCH FUNCTIONS
    # ============================================================================

    async def hybrid_search(
        self, 
        query: str, 
        filters: Optional[Dict] = None,
        boost_recent: bool = True,
        top_k: int = None
    ) -> List[Dict]:
        """
        Advanced hybrid search with recency boosting and confidence filtering
        """
        try:
            top_k = top_k or settings.VECTOR_SEARCH_TOP_K
            
            # Get semantic results
            semantic_results = await self.semantic_search(query, filters, top_k * 2)
            
            # Apply additional scoring
            scored_results = []
            for result in semantic_results:
                score = result["score"]
                
                # Boost score based on confidence if available
                confidence = result["metadata"].get("confidence", 0.8)
                if isinstance(confidence, (int, float)):
                    score *= (0.8 + confidence * 0.2)  # Boost high-confidence chunks
                
                # Boost recent documents if enabled
                if boost_recent:
                    # This would require timestamp in metadata
                    pass  # Implementation depends on metadata structure
                
                scored_results.append({
                    **result,
                    "final_score": score
                })
            
            # Re-sort by final score and limit results
            scored_results.sort(key=lambda x: x["final_score"], reverse=True)
            
            return scored_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return await self.semantic_search(query, filters, top_k)  # Fallback

    async def search_by_document_similarity(self, document_id: str, top_k: int = 5) -> List[Dict]:
        """Find documents similar to a given document"""
        try:
            # Get all chunks for the source document
            doc_chunks = self.index.query(
                vector=[0.0] * 1536,
                filter={"document_id": {"$eq": document_id}},
                top_k=1000,
                include_metadata=True,
                include_values=True
            )
            
            if not doc_chunks.matches:
                return []
            
            # Use the first chunk as representative vector
            representative_vector = doc_chunks.matches[0].values
            
            # Search for similar vectors from different documents
            similar_results = self.index.query(
                vector=representative_vector,
                filter={"document_id": {"$ne": document_id}},
                top_k=top_k * 3,  # Get more to account for multiple chunks per doc
                include_metadata=True
            )
            
            # Group by document and get best score per document
            doc_scores = {}
            for match in similar_results.matches:
                other_doc_id = match.metadata.get("document_id")
                if other_doc_id and other_doc_id != document_id:
                    if other_doc_id not in doc_scores or match.score > doc_scores[other_doc_id]["score"]:
                        doc_scores[other_doc_id] = {
                            "document_id": other_doc_id,
                            "score": match.score,
                            "metadata": match.metadata
                        }
            
            # Convert to list and sort
            similar_docs = list(doc_scores.values())
            similar_docs.sort(key=lambda x: x["score"], reverse=True)
            
            return similar_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    # ============================================================================
    # BULK OPERATIONS
    # ============================================================================

    async def bulk_update_metadata(self, updates: List[Dict]) -> Dict[str, int]:
        """Bulk update metadata for multiple vectors"""
        try:
            success_count = 0
            error_count = 0
            
            for update in updates:
                try:
                    vector_id = update.get("vector_id")
                    metadata_updates = update.get("metadata", {})
                    
                    if vector_id and metadata_updates:
                        self.index.update(
                            id=vector_id,
                            set_metadata=metadata_updates
                        )
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"Error in bulk update for vector {vector_id}: {e}")
                    error_count += 1
            
            logger.info(f"Bulk update completed: {success_count} success, {error_count} errors")
            return {"success": success_count, "errors": error_count}
            
        except Exception as e:
            logger.error(f"Error in bulk metadata update: {e}")
            return {"success": 0, "errors": len(updates)}

    async def export_vectors_metadata(self, document_ids: Optional[List[str]] = None) -> List[Dict]:
        """Export vector metadata for backup or analysis"""
        try:
            if document_ids:
                # Export specific documents
                filter_query = {"document_id": {"$in": document_ids}}
            else:
                # Export all vectors
                filter_query = {}
            
            query_result = self.index.query(
                vector=[0.0] * 1536,
                filter=filter_query,
                top_k=10000,
                include_metadata=True,
                include_values=False
            )
            
            exported_data = []
            for match in query_result.matches:
                exported_data.append({
                    "vector_id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            logger.info(f"Exported metadata for {len(exported_data)} vectors")
            return exported_data
            
        except Exception as e:
            logger.error(f"Error exporting vector metadata: {e}")
            return []

    # ============================================================================
    # MAINTENANCE FUNCTIONS
    # ============================================================================

    async def cleanup_orphaned_vectors(self, valid_document_ids: List[str]) -> int:
        """Remove vectors for documents that no longer exist in the database"""
        try:
            # Get all vectors
            all_vectors = self.index.query(
                vector=[0.0] * 1536,
                filter={},
                top_k=10000,
                include_metadata=True,
                include_values=False
            )
            
            # Find orphaned vectors
            orphaned_vector_ids = []
            for match in all_vectors.matches:
                doc_id = match.metadata.get("document_id")
                if doc_id and doc_id not in valid_document_ids:
                    orphaned_vector_ids.append(match.id)
            
            # Delete orphaned vectors in batches
            batch_size = 1000
            deleted_count = 0
            
            for i in range(0, len(orphaned_vector_ids), batch_size):
                batch = orphaned_vector_ids[i:i + batch_size]
                self.index.delete(ids=batch)
                deleted_count += len(batch)
            
            logger.info(f"Cleaned up {deleted_count} orphaned vectors")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned vectors: {e}")
            return 0

    async def reindex_document(self, document_id: str, chunks: List[Dict]) -> bool:
        """Reindex a document (delete old vectors and create new ones)"""
        try:
            # Delete existing vectors for this document
            await self.delete_document(document_id)
            
            # Wait a moment for deletion to propagate
            await asyncio.sleep(1)
            
            # Store new vectors
            return await self.store_document_chunks(document_id, chunks)
            
        except Exception as e:
            logger.error(f"Error reindexing document {document_id}: {e}")
            return False

    # ============================================================================
    # DEBUGGING AND UTILITIES
    # ============================================================================

    async def debug_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Debug search function with detailed information"""
        try:
            # Generate embedding
            query_embedding = await self.generate_embedding(query)
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Prepare debug information
            debug_info = {
                "query": query,
                "query_embedding_length": len(query_embedding),
                "results_count": len(search_results.matches),
                "results": []
            }
            
            for i, match in enumerate(search_results.matches):
                result_info = {
                    "rank": i + 1,
                    "vector_id": match.id,
                    "similarity_score": float(match.score),
                    "metadata_keys": list(match.metadata.keys()),
                    "document_id": match.metadata.get("document_id"),
                    "text_preview": match.metadata.get("text_preview", "")[:100]
                }
                debug_info["results"].append(result_info)
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error in debug search: {e}")
            return {"error": str(e)}

    def get_connection_info(self) -> Dict[str, str]:
        """Get connection information for debugging"""
        return {
            "index_name": self.index_name,
            "environment": self.environment,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "api_key_configured": bool(self.pinecone_api_key),
            "openai_key_configured": bool(self.openai_api_key)
        }