# ai_services.py - AI Services Manager (OpenAI Integration)
import asyncio
from typing import Dict, List, Optional, Any
import logging
import openai
import json
from config import settings

logger = logging.getLogger(__name__)

class AIService:
    """
    AI services manager that handles all OpenAI interactions
    Combines functionality from your auto_tagger.py, metadata_generator.py, and rag_engine.py
    """
    
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.model_name = settings.AI_MODEL_NAME
        self.temperature = settings.AI_TEMPERATURE
        
        logger.info(f"Initialized AI service with model: {self.model_name}")

    # ============================================================================
    # USE CASE DETECTION (Conversational Guidance)
    # ============================================================================

    async def detect_use_case(self, query: str, sector: str) -> str:
        """
        Intelligently detect the best use case for a query
        This enables the conversational guidance feature for your frontend
        """
        try:
            prompt = f"""
            Analyze this user query and suggest the most appropriate use case for the {sector} sector.
            
            Available use cases:
            1. Quick Playbook Answers - Direct questions about processes, guidelines, standards, "what does the playbook say about..."
            2. Lessons Learned - Learning from past projects, "what have we learned about...", experiences, insights
            3. Project Review / MOT - Health checks, status reviews, assessments, "review this project"
            4. TRL / RIRL Mapping - Technology readiness assessments, "what TRL/RIRL level is this..."
            5. Project Similarity - Finding similar past projects, "are there similar projects...", comparisons
            6. Change Management - Transitions, handovers, organizational changes, "how do we transition..."
            7. Product Acceptance - Approval processes, compliance, governance, "what's the approval process..."
            
            User query: "{query}"
            Sector: {sector}
            
            Analyze the intent and return ONLY the exact use case name that best matches.
            If unclear, default to "Quick Playbook Answers".
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            detected_use_case = response.choices[0].message.content.strip()
            
            # Validate against known use cases
            valid_use_cases = [
                "Quick Playbook Answers", "Lessons Learned", "Project Review / MOT",
                "TRL / RIRL Mapping", "Project Similarity", "Change Management", "Product Acceptance"
            ]
            
            if detected_use_case in valid_use_cases:
                logger.info(f"Detected use case '{detected_use_case}' for query: '{query[:50]}...'")
                return detected_use_case
            else:
                logger.warning(f"Invalid use case detected: '{detected_use_case}', defaulting to 'Quick Playbook Answers'")
                return "Quick Playbook Answers"
                
        except Exception as e:
            logger.error(f"Error detecting use case: {e}")
            return "Quick Playbook Answers"  # Safe default

    # ============================================================================
    # METADATA GENERATION (Document Processing)
    # ============================================================================

    async def generate_chunk_metadata(
        self, 
        text: str, 
        sector: str,
        use_case: Optional[str],
        suggested_title: Optional[str],
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a text chunk
        Replicates your metadata_generator.py functionality
        """
        try:
            prompt = f"""
            Analyze this text chunk and generate structured metadata for a strategic document management system.
            
            TEXT CHUNK:
            {text[:2000]}...
            
            Context Information:
            - Sector: {sector}
            - Use Case: {use_case or "General"}
            - Suggested Title: {suggested_title or "Unknown"}
            - Source URL: {source_url or "N/A"}
            
            Generate the following metadata fields in this exact format:
            
            TITLE: [Descriptive title for the source document - 5-8 words]
            TOPIC: [Main topic/area covered - e.g., governance, technology, planning]
            TAGS: [5-7 comma-separated keywords relevant to the content]
            SUMMARY: [2-sentence summary of what this chunk covers]
            CONFIDENCE: [0.0-1.0 confidence score for the quality of this chunk]
            SECTOR_RELEVANCE: [How relevant this is to the {sector} sector: high/medium/low]
            
            Be specific and accurate. Focus on strategic and operational content.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            
            # Parse the structured response
            content = response.choices[0].message.content
            metadata = self._parse_metadata_response(content)
            
            # Add context information
            metadata.update({
                "title": metadata.get("title", suggested_title or "Untitled"),
                "sector": sector,
                "use_case": use_case or "General",
                "source_url": source_url or "N/A",
                "ai_generated": True,
                "chunk_length": len(text)
            })
            
            logger.debug(f"Generated metadata for chunk: {metadata.get('title', 'Untitled')}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating chunk metadata: {e}")
            # Return basic metadata on error
            return {
                "title": suggested_title or "Untitled",
                "topic": "General",
                "tags": "document, content",
                "summary": f"Content from {sector} sector document.",
                "confidence": 0.5,
                "sector": sector,
                "use_case": use_case or "General",
                "source_url": source_url or "N/A",
                "ai_generated": True,
                "chunk_length": len(text)
            }

    def _parse_metadata_response(self, content: str) -> Dict[str, str]:
        """Parse structured metadata response from AI"""
        metadata = {}
        
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                clean_key = key.strip().lower().replace(' ', '_')
                clean_value = value.strip()
                
                if clean_value:
                    metadata[clean_key] = clean_value
        
        return metadata

    # ============================================================================
    # RAG RESPONSE GENERATION (Main Chat Function)
    # ============================================================================

    async def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        prompt_template: Optional[str],
        user_type: str
    ) -> str:
        """
        Generate AI response using retrieved context
        This is the main function called by the chat endpoint
        """
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(context_docs)
            
            # Choose prompt template
            if prompt_template:
                # Use custom prompt template from database
                formatted_prompt = self._format_custom_prompt(
                    prompt_template, context, query, user_type
                )
            else:
                # Use default prompt based on user type
                formatted_prompt = self._create_default_prompt(context, query, user_type)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.temperature,
                max_tokens=1200
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            logger.info(f"Generated response for {user_type} user, query: '{query[:50]}...'")
            return generated_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

    def _prepare_context(self, context_docs: List[Dict]) -> str:
        """Prepare context string from retrieved documents"""
        if not context_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(context_docs[:5]):  # Limit to top 5 for token management
            doc_title = doc.get("metadata", {}).get("title", f"Document {i+1}")
            doc_text = doc.get("text", "")
            
            context_parts.append(f"Document {i+1} - {doc_title}:\n{doc_text}\n")
        
        return "\n".join(context_parts)

    def _format_custom_prompt(
        self, 
        template: str, 
        context: str, 
        query: str, 
        user_type: str
    ) -> str:
        """Format custom prompt template with variables"""
        try:
            return template.format(
                context=context,
                question=query,
                query=query,
                user_type=user_type
            )
        except KeyError as e:
            logger.warning(f"Custom prompt template missing variable {e}, using fallback")
            return self._create_default_prompt(context, query, user_type)

    def _create_default_prompt(self, context: str, query: str, user_type: str) -> str:
        """Create default prompt based on user type"""
        
        user_instructions = {
            "admin": "You are assisting an administrator. Provide detailed, technical insights with governance implications.",
            "analyst": "You are assisting an analyst. Provide data-driven insights with specific metrics and strategic recommendations.",
            "public": "You are assisting a general user. Provide clear, accessible explanations with practical guidance."
        }
        
        instruction = user_instructions.get(user_type, user_instructions["public"])
        
        return f"""
        {instruction}
        
        Based on the following context documents, answer the user's question.
        
        Guidelines:
        - Provide specific, actionable insights
        - Reference the source documents when possible
        - If the answer isn't clearly in the context, acknowledge this
        - Focus on strategic and operational implications
        - Be concise but comprehensive
        
        CONTEXT DOCUMENTS:
        {context}
        
        USER QUESTION: {query}
        
        RESPONSE:
        """

    # ============================================================================
    # SPECIALIZED USE CASE RESPONSES
    # ============================================================================

    async def generate_use_case_response(
        self,
        query: str,
        context_docs: List[Dict],
        use_case: str,
        sector: str
    ) -> str:
        """Generate specialized response based on use case"""
        
        use_case_prompts = {
            "Quick Playbook Answers": self._get_playbook_prompt(),
            "Lessons Learned": self._get_lessons_learned_prompt(),
            "Project Review / MOT": self._get_project_review_prompt(),
            "TRL / RIRL Mapping": self._get_readiness_assessment_prompt(),
            "Project Similarity": self._get_similarity_prompt(),
            "Change Management": self._get_change_management_prompt(),
            "Product Acceptance": self._get_acceptance_prompt()
        }
        
        prompt_template = use_case_prompts.get(use_case, self._get_default_prompt())
        context = self._prepare_context(context_docs)
        
        formatted_prompt = prompt_template.format(
            context=context,
            question=query,
            sector=sector
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating use case response: {e}")
            return await self.generate_response(query, context_docs, None, "public")

    def _get_playbook_prompt(self) -> str:
        return """
        You are an AI assistant that provides direct answers from project management playbooks and guidelines.
        
        Answer the question based on the following context documents.
        Be specific and reference exact procedures, guidelines, or standards mentioned.
        If specific guidance isn't available, clearly state this.
        
        Context: {context}
        Question: {question}
        
        Response (focus on direct, actionable guidance):
        """

    def _get_lessons_learned_prompt(self) -> str:
        return """
        You are an AI assistant that summarizes lessons learned from past projects.
        
        Based on the context, identify and summarize relevant lessons learned that address the question.
        Organize insights as key findings with specific examples where available.
        Focus on what worked, what didn't, and recommendations for future projects.
        
        Context: {context}
        Question: {question}
        
        Lessons Learned Summary:
        """

    def _get_project_review_prompt(self) -> str:
        return """
        You are an AI assistant that helps with project health checks and reviews.
        
        Based on the context, provide a structured assessment addressing the question.
        Structure your response with:
        - Current Status Assessment
        - Key Risks and Issues
        - Recommendations for Action
        - Next Steps
        
        Context: {context}
        Project/Question: {question}
        
        Project Assessment:
        """

    def _get_readiness_assessment_prompt(self) -> str:
        return """
        You are an AI assistant that helps assess Technology Readiness Levels (TRL) and R&D Innovation Readiness Levels (RIRL).
        
        Based on the context, assess the readiness level for the technology or project mentioned.
        Provide:
        - Current readiness level assessment (TRL 1-9 or RIRL equivalent)
        - Justification for the assessment
        - Key criteria met and missing
        - Steps to reach the next level
        
        Context: {context}
        Question: {question}
        
        Readiness Assessment:
        """

    def _get_similarity_prompt(self) -> str:
        return """
        You are an AI assistant that identifies similar projects and draws comparisons.
        
        Based on the context, identify projects or initiatives similar to what's described in the question.
        For each similar project, provide:
        - Project name and brief description
        - Key similarities
        - Relevant lessons or outcomes
        - Applicability to the current situation
        
        Context: {context}
        Question: {question}
        
        Similar Projects Analysis:
        """

    def _get_change_management_prompt(self) -> str:
        return """
        You are an AI assistant that helps with project transitions and change management.
        
        Based on the context, create guidance for the transition or change described in the question.
        Organize your response as a structured approach covering:
        - Key stakeholders and their roles
        - Critical transition activities
        - Documentation and knowledge transfer needs
        - Risk mitigation strategies
        - Success criteria
        
        Context: {context}
        Question: {question}
        
        Change Management Plan:
        """

    def _get_acceptance_prompt(self) -> str:
        return """
        You are an AI assistant that helps with product acceptance and approval processes.
        
        Based on the context, explain the relevant acceptance criteria, approval processes, or governance requirements.
        Include:
        - Specific requirements and criteria
        - Approval stages and stakeholders
        - Documentation needed
        - Compliance considerations
        - Timeline and process steps
        
        Context: {context}
        Question: {question}
        
        Acceptance Process Guidance:
        """

    def _get_default_prompt(self) -> str:
        return """
        You are an AI assistant helping with strategic planning and project management in the {sector} sector.
        
        Based on the context documents, provide a comprehensive answer to the question.
        Be specific, actionable, and reference the source material when relevant.
        
        Context: {context}
        Question: {question}
        
        Response:
        """

    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================

    def calculate_confidence(self, docs: List[Dict]) -> float:
        """Calculate confidence score based on document relevance and quality"""
        if not docs:
            return 0.0
        
        # Average of top 3 document scores, weighted by metadata confidence
        top_docs = docs[:3]
        scores = []
        
        for doc in top_docs:
            base_score = doc.get("score", 0.0)
            
            # Boost score based on metadata confidence if available
            metadata_confidence = doc.get("metadata", {}).get("confidence", 0.8)
            if isinstance(metadata_confidence, (int, float)):
                adjusted_score = base_score * (0.7 + metadata_confidence * 0.3)
            else:
                adjusted_score = base_score * 0.85
            
            scores.append(adjusted_score)
        
        return sum(scores) / len(scores) if scores else 0.0

    async def validate_api_key(self) -> bool:
        """Validate OpenAI API key"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {e}")
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """Get service configuration information"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_key_configured": bool(self.openai_api_key),
            "embedding_model": settings.EMBEDDING_MODEL_NAME
        }