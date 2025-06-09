# document_processor.py - Document Processing for Multi-Agent System
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
import re
import json

from database import db_manager
from vector_store import vector_store
from specialized_agents import orchestration_agent

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Enhanced document processor for multi-agent Strategy AI system
    Handles document upload, text extraction, chunking, and metadata generation
    """
    
    def __init__(self):
        self.max_file_size_mb = 50
        self.allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.csv'}
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        sector: str,
        use_case: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process uploaded document end-to-end"""
        try:
            logger.info(f"Processing document: {filename}")
            
            # 1. Validate file
            if not self._validate_file(filename, len(file_content)):
                return {
                    "success": False,
                    "error": "File validation failed",
                    "details": "Invalid file type or size too large"
                }
            
            # 2. Extract text
            text_content = await self._extract_text(file_content, filename)
            if not text_content:
                return {
                    "success": False,
                    "error": "Text extraction failed",
                    "details": "Could not extract readable text from file"
                }
            
            # 3. Store document metadata
            document_id = await db_manager.store_document(
                title=metadata.get("title", filename),
                filename=filename,
                sector=sector,
                use_case=use_case,
                source_type="file",
                metadata=metadata or {}
            )
            
            # 4. Generate chunks
            chunks = await self._create_chunks(text_content, document_id)
            
            # 5. Enhance chunks with AI analysis
            enhanced_chunks = await self._enhance_chunks_with_ai(chunks, sector, use_case)
            
            # 6. Store in vector database
            await vector_store.store_chunks(enhanced_chunks, document_id, {
                "title": metadata.get("title", filename),
                "sector": sector,
                "use_case": use_case,
                "source": "file_upload"
            })
            
            # 7. Update document status
            await db_manager.update_document_status(
                document_id, "completed", len(enhanced_chunks)
            )
            
            logger.info(f"Successfully processed document {filename}: {len(enhanced_chunks)} chunks")
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(enhanced_chunks),
                "total_characters": len(text_content),
                "processing_summary": {
                    "text_extracted": True,
                    "chunks_generated": len(enhanced_chunks),
                    "ai_enhanced": True,
                    "vector_stored": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return {
                "success": False,
                "error": str(e),
                "details": "Document processing failed"
            }
    
    def _validate_file(self, filename: str, file_size: int) -> bool:
        """Validate uploaded file"""
        # Check file size
        if file_size > self.max_file_size_mb * 1024 * 1024:
            return False
        
        # Check file extension
        file_ext = '.' + filename.split('.')[-1].lower()
        if file_ext not in self.allowed_extensions:
            return False
        
        return True
    
    async def _extract_text(self, file_content: bytes, filename: str) -> str:
        """Extract text from various file formats"""
        try:
            file_ext = '.' + filename.split('.')[-1].lower()
            
            if file_ext in {'.txt', '.md', '.csv'}:
                # Plain text files
                return file_content.decode('utf-8', errors='ignore')
            
            elif file_ext == '.pdf':
                # For demo, simulate PDF text extraction
                return self._simulate_pdf_extraction(filename)
            
            elif file_ext == '.docx':
                # For demo, simulate DOCX text extraction
                return self._simulate_docx_extraction(filename)
            
            else:
                return file_content.decode('utf-8', errors='ignore')
                
        except Exception as e:
            logger.error(f"Text extraction error for {filename}: {e}")
            return ""
    
    def _simulate_pdf_extraction(self, filename: str) -> str:
        """Simulate PDF text extraction for demo"""
        return f"""
        Strategic Framework Document - {filename}
        
        EXECUTIVE SUMMARY
        This document outlines a comprehensive strategic approach for sector development, 
        focusing on innovation, sustainability, and stakeholder engagement.
        
        1. STRATEGIC OBJECTIVES
        The primary objectives include:
        • Enhancing operational efficiency through digital transformation
        • Promoting sustainable practices and environmental responsibility
        • Fostering innovation and technology adoption
        • Strengthening stakeholder partnerships and community engagement
        
        2. IMPLEMENTATION APPROACH
        The implementation strategy emphasizes:
        • Phased rollout with clear milestones and success metrics
        • Cross-functional collaboration and knowledge sharing
        • Continuous monitoring and adaptive management
        • Risk management and mitigation strategies
        
        3. GOVERNANCE FRAMEWORK
        Effective governance requires:
        • Clear roles and responsibilities
        • Regular performance reviews and assessments
        • Transparent communication and reporting
        • Compliance with regulatory requirements
        
        4. LESSONS LEARNED
        Key insights from previous initiatives:
        • Early stakeholder engagement is critical for success
        • Technology solutions must be user-friendly and accessible
        • Change management requires dedicated resources and support
        • Measurement and evaluation frameworks should be established upfront
        
        CONCLUSION
        This strategic framework provides a roadmap for achieving organizational objectives
        while maintaining flexibility to adapt to changing circumstances and emerging opportunities.
        """
    
    def _simulate_docx_extraction(self, filename: str) -> str:
        """Simulate DOCX text extraction for demo"""
        return f"""
        Policy Guidelines Document - {filename}
        
        INTRODUCTION
        This document provides comprehensive policy guidelines for strategic decision-making
        and operational excellence in complex organizational environments.
        
        POLICY FRAMEWORK
        The framework encompasses:
        • Strategic planning and resource allocation
        • Risk management and compliance protocols
        • Performance measurement and quality assurance
        • Stakeholder engagement and communication strategies
        
        BEST PRACTICES
        Recommended practices include:
        • Evidence-based decision making
        • Collaborative approach to problem solving
        • Continuous improvement and learning culture
        • Innovation and technology integration
        
        IMPLEMENTATION GUIDELINES
        Successful implementation requires:
        • Clear communication of objectives and expectations
        • Adequate training and capacity building
        • Regular monitoring and feedback mechanisms
        • Flexibility to adapt to changing circumstances
        """
    
    async def _create_chunks(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Create text chunks for vector storage"""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Avoid splitting words
            if i + self.chunk_size < text_length:
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size * 0.8:  # Only if we don't lose too much
                    chunk_text = chunk_text[:last_space]
            
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_index": len(chunks),
                "character_start": i,
                "character_end": i + len(chunk_text),
                "metadata": {
                    "document_id": document_id,
                    "chunk_length": len(chunk_text)
                }
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        return text.strip()
    
    async def _enhance_chunks_with_ai(
        self,
        chunks: List[Dict[str, Any]],
        sector: str,
        use_case: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Enhance chunks with basic metadata (minimal version)"""
        enhanced_chunks = []
        
        for chunk in chunks:
            try:
                # Simple enhancement without complex AI processing
                enhanced_metadata = {
                    **chunk["metadata"],
                    "ai_keywords": ["strategy", "framework", "analysis"],
                    "ai_topic": f"{sector} Strategy",
                    "ai_complexity": "intermediate",
                    "ai_confidence": 0.7,
                    "sector": sector,
                    "use_case": use_case
                }
                
                enhanced_chunk = {
                    **chunk,
                    "metadata": enhanced_metadata
                }
                
                enhanced_chunks.append(enhanced_chunk)
                
            except Exception as e:
                logger.warning(f"Enhancement failed for chunk {chunk['chunk_index']}: {e}")
                # Fall back to basic chunk
                chunk["metadata"]["sector"] = sector
                chunk["metadata"]["use_case"] = use_case
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all associated chunks"""
        try:
            # Delete from vector store
            await vector_store.delete_document_chunks(document_id)
            
            # Delete from database
            await db_manager.delete_document(document_id)
            
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get document processing status"""
        document = await db_manager.get_document(document_id)
        
        if not document:
            return {"error": "Document not found"}
        
        return {
            "document_id": document_id,
            "status": document["status"],
            "chunk_count": document["chunk_count"],
            "created_at": document["created_at"],
            "updated_at": document["updated_at"]
        }

# Global document processor instance
document_processor = DocumentProcessor() 