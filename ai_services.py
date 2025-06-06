# ai_services.py - AI Services Manager (OpenAI Integration)
import asyncio
from typing import Dict, List, Optional, Any
import logging
import openai
import json
from config import Settings

logger = logging.getLogger(__name__)

class AIService:
    """
    AI services manager that handles all OpenAI interactions
    """
    
    def __init__(self):
        self.settings = Settings()
        self.openai_api_key = getattr(self.settings, 'OPENAI_API_KEY', None)
        
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set, using demo mode")
            self.demo_mode = True
            return
        
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.model_name = getattr(self.settings, 'AI_MODEL_NAME', 'gpt-3.5-turbo')
        self.temperature = getattr(self.settings, 'AI_TEMPERATURE', 0.7)
        self.demo_mode = False
        
        logger.info(f"Initialized AI service with model: {self.model_name}")

    async def detect_use_case(self, query: str, sector: str) -> str:
        """
        Intelligently detect the best use case for a query
        """
        if self.demo_mode:
            return f"{sector} Analysis" if sector != "General" else "Quick Playbook Answers"
        
        try:
            prompt = f"""
            Analyze this user query and suggest the most appropriate use case for the {sector} sector.
            
            Available use cases:
            1. Quick Playbook Answers - Direct questions about processes, guidelines, standards
            2. Lessons Learned - Learning from past projects, experiences, insights
            3. Project Review / MOT - Health checks, status reviews, assessments
            4. TRL / RIRL Mapping - Technology readiness assessments
            5. Project Similarity - Finding similar past projects, comparisons
            6. Change Management - Transitions, handovers, organizational changes
            7. Product Acceptance - Approval processes, compliance, governance
            
            User query: "{query}"
            Sector: {sector}
            
            Return ONLY the exact use case name that best matches.
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
                return detected_use_case
            else:
                return "Quick Playbook Answers"
                
        except Exception as e:
            logger.error(f"Error detecting use case: {e}")
            return "Quick Playbook Answers"

    async def generate_response(
        self,
        query: str,
        sector: str = "General",
        use_case: Optional[str] = None,
        user_type: str = "public"
    ) -> Dict[str, Any]:
        """
        Generate AI response for chat
        """
        if self.demo_mode:
            return {
                "response": f"[DEMO MODE] I received your message about '{query[:50]}...' in the {sector} sector. To enable real AI responses, please set your OPENAI_API_KEY environment variable.",
                "confidence": 0.5,
                "sources": [{"title": "Demo Mode Active", "relevance": 1.0}],
                "suggested_use_case": use_case or f"{sector} Analysis" if sector != "General" else None
            }
        
        try:
            # Detect use case if not provided
            if not use_case:
                use_case = await self.detect_use_case(query, sector)
            
            # Create specialized prompt based on sector and use case
            prompt = self._create_sector_prompt(query, sector, use_case, user_type)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content
            
            # Calculate confidence based on response quality
            confidence = self._calculate_response_confidence(ai_response, query)
            
            return {
                "response": ai_response,
                "confidence": confidence,
                "sources": self._generate_mock_sources(sector, use_case),
                "suggested_use_case": use_case
            }
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return {
                "response": f"I apologize, but I'm experiencing technical difficulties. Please try again later. (Error: {str(e)})",
                "confidence": 0.0,
                "sources": [],
                "suggested_use_case": use_case
            }

    def _create_sector_prompt(self, query: str, sector: str, use_case: str, user_type: str) -> str:
        """Create specialized prompt based on sector and use case"""
        
        base_context = f"""
        You are an expert AI assistant specializing in {sector} sector strategy and policy analysis.
        You are helping a {user_type} user with a {use_case} request.
        
        Provide comprehensive, actionable insights while being:
        - Professional and authoritative
        - Specific to the {sector} sector
        - Focused on {use_case} methodology
        - Practical and implementation-focused
        """
        
        if sector.lower() == "transport":
            sector_context = """
            You have deep expertise in:
            - Transport policy and strategy development
            - Infrastructure planning and delivery
            - Connected and autonomous vehicles
            - Urban mobility and smart cities
            - Rail, road, aviation, and maritime sectors
            - Regulatory frameworks and governance
            - Technology readiness and innovation adoption
            """
        elif sector.lower() == "energy":
            sector_context = """
            You have deep expertise in:
            - Energy policy and transition strategies
            - Renewable energy deployment
            - Grid modernization and storage
            - Energy efficiency and demand management
            - Regulatory frameworks and market design
            - Clean technology innovation
            """
        else:
            sector_context = f"""
            You have broad expertise in {sector} sector strategy, policy development,
            technology adoption, and organizational transformation.
            """
        
        use_case_guidance = self._get_use_case_guidance(use_case)
        
        full_prompt = f"""
        {base_context}
        
        {sector_context}
        
        {use_case_guidance}
        
        User Question: {query}
        
        Please provide a detailed, helpful response that addresses their specific needs.
        """
        
        return full_prompt

    def _get_use_case_guidance(self, use_case: str) -> str:
        """Get specific guidance based on use case"""
        guidance_map = {
            "Quick Playbook Answers": "Focus on providing direct, actionable guidance from best practices and established frameworks.",
            "Lessons Learned": "Emphasize insights from past projects, what worked well, what didn't, and key takeaways.",
            "Project Review / MOT": "Provide structured assessment criteria, health check frameworks, and evaluation methodologies.",
            "TRL / RIRL Mapping": "Focus on technology and innovation readiness levels, maturity assessments, and scaling considerations.",
            "Project Similarity": "Compare and contrast with similar initiatives, highlighting relevant parallels and differences.",
            "Change Management": "Address organizational transformation, stakeholder engagement, and transition strategies.",
            "Product Acceptance": "Focus on approval processes, compliance requirements, and governance frameworks."
        }
        
        return guidance_map.get(use_case, "Provide comprehensive strategic guidance.")

    def _calculate_response_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score based on response quality"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on response length and detail
        if len(response) < 100:
            confidence -= 0.2
        elif len(response) > 500:
            confidence += 0.1
        
        # Check for specific keywords that indicate good responses
        quality_indicators = [
            "strategy", "framework", "approach", "recommend", "suggest",
            "analysis", "assessment", "evaluation", "implementation",
            "best practice", "guidance", "methodology"
        ]
        
        found_indicators = sum(1 for indicator in quality_indicators 
                             if indicator.lower() in response.lower())
        
        confidence += min(found_indicators * 0.02, 0.1)
        
        return min(confidence, 0.95)  # Cap at 95%

    def _generate_mock_sources(self, sector: str, use_case: str) -> List[Dict[str, Any]]:
        """Generate realistic mock sources based on sector and use case"""
        sources = []
        
        if sector.lower() == "transport":
            sources.extend([
                {"title": f"{sector} Strategy Framework 2024", "relevance": 0.92},
                {"title": f"Connected Places Catapult {use_case} Guide", "relevance": 0.88},
                {"title": f"UK {sector} Innovation Roadmap", "relevance": 0.85}
            ])
        else:
            sources.extend([
                {"title": f"{sector} Policy Guidelines", "relevance": 0.90},
                {"title": f"Strategic {use_case} Framework", "relevance": 0.87},
                {"title": f"{sector} Best Practices Compendium", "relevance": 0.84}
            ])
        
        return sources

    async def validate_api_key(self) -> bool:
        """Validate OpenAI API key"""
        if self.demo_mode:
            return False
        
        try:
            self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False

# Global AI service instance
ai_service = AIService() 