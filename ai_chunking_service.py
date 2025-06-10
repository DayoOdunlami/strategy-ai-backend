import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import tiktoken
import openai
import PyPDF2
from docx import Document
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChunkStrategy(Enum):
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    FIXED_TOKEN = "fixed_token"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"

@dataclass
class ChunkingConfig:
    strategy: ChunkStrategy
    target_tokens: int = 768  # Optimal for most embedding models
    max_tokens: int = 1024    # Hard limit
    overlap_tokens: int = 128  # Semantic overlap
    min_chunk_tokens: int = 100
    preserve_structure: bool = True
    generate_summaries: bool = True
    
class SmartChunkingService:
    """
    Production-ready AI chunking service optimized for Pinecone vector storage
    """
    
    def __init__(self):
        # Initialize OpenAI client (v0.28 format)
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize tokenizer (for OpenAI models)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Chunking configurations by use case
        self.configs = {
            "strategy": ChunkingConfig(
                strategy=ChunkStrategy.HIERARCHICAL,
                target_tokens=512,
                generate_summaries=True,
                preserve_structure=True
            ),
            "general": ChunkingConfig(
                strategy=ChunkStrategy.SEMANTIC,
                target_tokens=768,
                generate_summaries=False
            ),
            "technical": ChunkingConfig(
                strategy=ChunkStrategy.SEMANTIC,
                target_tokens=1024,
                overlap_tokens=200,
                preserve_structure=True
            ),
            "quick_playbook_answers": ChunkingConfig(
                strategy=ChunkStrategy.PARAGRAPH,
                target_tokens=512,
                preserve_structure=True
            ),
            "lessons_learned": ChunkingConfig(
                strategy=ChunkStrategy.HIERARCHICAL,
                target_tokens=768,
                generate_summaries=True
            )
        }

    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract clean text from various file formats"""
        try:
            if filename.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_content)
            elif filename.lower().endswith('.docx'):
                return self._extract_from_docx(file_content)
            elif filename.lower().endswith(('.txt', '.md')):
                return file_content.decode('utf-8', errors='ignore')
            else:
                # Fallback to text
                return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Text extraction error for {filename}: {e}")
            return file_content.decode('utf-8', errors='ignore')

    def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF with structure preservation"""
        try:
            import io
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise

    def _extract_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX with structure preservation"""
        try:
            import io
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    # Preserve heading levels
                    if paragraph.style.name.startswith('Heading'):
                        text += f"\n## {paragraph.text}\n"
                    else:
                        text += f"{paragraph.text}\n"
            
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better chunking"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common PDF artifacts
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Fix sentence spacing
        
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI tokenizer"""
        return len(self.tokenizer.encode(text))

    async def analyze_document_smart(self, content: str, filename: str, sector: str, use_case: str) -> Dict[str, Any]:
        """
        AI-powered document analysis optimized for Pinecone chunking
        """
        try:
            # Get configuration for use case
            config = self.configs.get(use_case.lower().replace(' ', '_'), self.configs["general"])
            
            # Basic content analysis
            word_count = len(content.split())
            token_count = self.count_tokens(content)
            
            # Structure detection
            has_headers = bool(re.search(r'^#+\s+\w+', content, re.MULTILINE) or 
                             re.search(r'(?:Chapter|Section|Part)\s+\d+', content, re.IGNORECASE))
            
            has_technical = any(term in content.lower() for term in [
                'implementation', 'specification', 'requirements', 'methodology',
                'algorithm', 'protocol', 'framework', 'architecture'
            ])
            
            has_data = any(term in content.lower() for term in [
                'table', 'figure', 'chart', 'data', 'statistics', 'results'
            ])
            
            # AI-powered content analysis
            ai_analysis = await self._get_ai_content_analysis(content[:2000], sector, use_case)
            
            # Determine optimal strategy
            if ai_analysis.get('complexity') == 'high' or has_technical:
                strategy = ChunkStrategy.HIERARCHICAL
                target_tokens = 1024
            elif has_headers and word_count > 2000:
                strategy = ChunkStrategy.SEMANTIC
                target_tokens = 768
            elif use_case.lower() in ['strategy', 'lessons_learned']:
                strategy = ChunkStrategy.HIERARCHICAL
                target_tokens = 512
            else:
                strategy = ChunkStrategy.SEMANTIC
                target_tokens = 768
            
            # Estimate chunks (Pinecone optimized)
            estimated_chunks = max(1, token_count // target_tokens)
            if strategy == ChunkStrategy.HIERARCHICAL:
                estimated_chunks = int(estimated_chunks * 1.3)  # Account for summaries
            
            return {
                "contentType": self._determine_content_type(filename),
                "complexity": ai_analysis.get('complexity', 'medium'),
                "recommendedChunking": {
                    "type": strategy.value,
                    "size": target_tokens,
                    "overlap": min(200, target_tokens // 4),
                    "strategy": f"AI-optimized {strategy.value} chunking for Pinecone"
                },
                "estimatedChunks": estimated_chunks,
                "aiInsights": {
                    "wordCount": word_count,
                    "tokenCount": token_count,
                    "hasStructure": has_headers,
                    "hasTechnicalContent": has_technical,
                    "hasDataElements": has_data,
                    "recommendedStrategy": f"Optimized for {sector} sector {use_case} use case",
                    "pineconeOptimized": True,
                    "targetEmbeddingSize": target_tokens,
                    **ai_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Smart analysis error: {e}")
            # Fallback to basic analysis
            return self._fallback_analysis(content, filename, sector, use_case)

    async def _get_ai_content_analysis(self, content_sample: str, sector: str, use_case: str) -> Dict[str, Any]:
        """Get AI analysis of content for optimal chunking strategy"""
        try:
            prompt = f"""
            Analyze this document content for optimal chunking strategy in the {sector} sector for {use_case} use case.
            
            Content sample:
            {content_sample}
            
            Provide analysis in this exact JSON format:
            {{
                "complexity": "low|medium|high",
                "primary_content_type": "technical|narrative|procedural|mixed",
                "structure_quality": "poor|good|excellent",
                "recommended_overlap": 50-300,
                "key_topics": ["topic1", "topic2", "topic3"],
                "chunking_priority": "context_preservation|information_density|retrieval_accuracy"
            }}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse JSON response
            import json
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {
                "complexity": "medium",
                "primary_content_type": "mixed",
                "structure_quality": "good",
                "recommended_overlap": 128,
                "key_topics": ["general"],
                "chunking_priority": "context_preservation"
            }

    def _determine_content_type(self, filename: str) -> str:
        """Determine content type from filename"""
        if filename.lower().endswith('.pdf'):
            return "PDF Document"
        elif filename.lower().endswith(('.doc', '.docx')):
            return "Word Document"
        elif filename.lower().endswith(('.txt', '.md')):
            return "Text Document"
        else:
            return "Document"

    def _fallback_analysis(self, content: str, filename: str, sector: str, use_case: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        token_count = self.count_tokens(content)
        return {
            "contentType": self._determine_content_type(filename),
            "complexity": "medium",
            "recommendedChunking": {
                "type": "semantic",
                "size": 768,
                "overlap": 128,
                "strategy": "Fallback semantic chunking"
            },
            "estimatedChunks": max(1, token_count // 768),
            "aiInsights": {
                "wordCount": len(content.split()),
                "tokenCount": token_count,
                "fallback": True,
                "pineconeOptimized": True
            }
        }

    async def chunk_document_for_pinecone(self, content: str, config: ChunkingConfig, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create Pinecone-optimized chunks with rich metadata
        """
        try:
            if config.strategy == ChunkStrategy.HIERARCHICAL:
                return await self._hierarchical_chunking(content, config, metadata)
            elif config.strategy == ChunkStrategy.SEMANTIC:
                return await self._semantic_chunking(content, config, metadata)
            else:
                return await self._token_based_chunking(content, config, metadata)
                
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            # Fallback to simple chunking
            return await self._token_based_chunking(content, config, metadata)

    async def _hierarchical_chunking(self, content: str, config: ChunkingConfig, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create hierarchical chunks: summaries + detailed chunks"""
        chunks = []
        
        # First pass: Create detailed chunks
        detailed_chunks = await self._semantic_chunking(content, config, metadata)
        
        # Second pass: Create summary chunks for every 3-5 detailed chunks
        for i in range(0, len(detailed_chunks), 4):
            chunk_group = detailed_chunks[i:i+4]
            combined_text = "\n\n".join([chunk["content"] for chunk in chunk_group])
            
            # Generate AI summary
            summary = await self._generate_chunk_summary(combined_text, metadata)
            
            # Add summary chunk
            summary_chunk = {
                "content": summary,
                "metadata": {
                    **metadata,
                    "chunk_type": "summary",
                    "chunk_index": len(chunks),
                    "detail_chunks": [chunk["metadata"]["chunk_index"] for chunk in chunk_group],
                    "token_count": self.count_tokens(summary)
                }
            }
            chunks.append(summary_chunk)
        
        # Add all detailed chunks
        for chunk in detailed_chunks:
            chunk["metadata"]["chunk_type"] = "detail"
            chunks.append(chunk)
        
        return chunks

    async def _semantic_chunking(self, content: str, config: ChunkingConfig, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantically coherent chunks using sentence embeddings"""
        # Split into sentences
        sentences = nltk.sent_tokenize(content)
        
        # Calculate embeddings for semantic similarity
        embeddings = self.sentence_model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed target
            if current_tokens + sentence_tokens > config.target_tokens and current_chunk:
                # Find optimal break point using semantic similarity
                chunk_text = " ".join(current_chunk)
                
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "token_count": current_tokens,
                        "sentence_count": len(current_chunk),
                        "chunk_type": "semantic"
                    }
                })
                
                # Start new chunk with overlap
                if config.overlap_tokens > 0:
                    overlap_sentences = self._get_semantic_overlap(current_chunk, config.overlap_tokens)
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "content": chunk_text.strip(),
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "token_count": current_tokens,
                    "sentence_count": len(current_chunk),
                    "chunk_type": "semantic"
                }
            })
        
        return chunks

    async def _token_based_chunking(self, content: str, config: ChunkingConfig, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple token-based chunking with smart boundaries"""
        words = content.split()
        chunks = []
        chunk_index = 0
        
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            
            if current_tokens + word_tokens > config.target_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "token_count": current_tokens,
                        "word_count": len(current_chunk),
                        "chunk_type": "token_based"
                    }
                })
                
                # Add overlap
                if config.overlap_tokens > 0:
                    overlap_words = current_chunk[-config.overlap_tokens//4:]  # Approx overlap
                    current_chunk = overlap_words + [word]
                    current_tokens = sum(self.count_tokens(w + " ") for w in current_chunk)
                else:
                    current_chunk = [word]
                    current_tokens = word_tokens
                
                chunk_index += 1
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "content": chunk_text.strip(),
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "token_count": current_tokens,
                    "word_count": len(current_chunk),
                    "chunk_type": "token_based"
                }
            })
        
        return chunks

    def _get_semantic_overlap(self, sentences: List[str], target_tokens: int) -> List[str]:
        """Get semantically relevant overlap sentences"""
        overlap_sentences = []
        overlap_tokens = 0
        
        # Take last few sentences up to target tokens
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= target_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences

    async def _generate_chunk_summary(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate AI summary for chunk group"""
        try:
            sector = metadata.get('sector', 'General')
            use_case = metadata.get('use_case', 'general')
            
            prompt = f"""
            Create a concise summary (max 200 words) of this content for the {sector} sector {use_case} use case.
            Focus on key insights and actionable information.
            
            Content:
            {content[:1500]}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=250
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            # Fallback: extract first paragraph
            paragraphs = content.split('\n\n')
            return paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0]


# Global service instance
chunking_service = SmartChunkingService() 