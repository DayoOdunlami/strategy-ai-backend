# ai_services.py - Multi-Model AI Services Manager (OpenAI + Claude)
import asyncio
from typing import Dict, List, Optional, Any
import logging
import json
from config import Settings

logger = logging.getLogger(__name__)

class AIService:
    """
    Multi-model AI services manager that supports OpenAI and Claude
    """
    
    def __init__(self):
        self.settings = Settings()
        self.demo_mode = True
        self.available_models = []
        
        # Try to initialize OpenAI
        self._init_openai()
        
        # Try to initialize Claude
        self._init_claude()
        
        # Set default model
        if self.available_models:
            self.demo_mode = False
            self.current_model = self.available_models[0]
            logger.info(f"AI service initialized with models: {self.available_models}")
        else:
            logger.warning("No AI models available, using demo mode")
            self.current_model = "demo"

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            
            openai_key = getattr(self.settings, 'OPENAI_API_KEY', None)
            if openai_key:
                # OpenAI 0.28.1 uses different syntax
                openai.api_key = openai_key
                
                # Test the connection
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                
                self.available_models.append("openai")
                self.openai = openai
                logger.info("OpenAI initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")

    def _init_claude(self):
        """Initialize Claude (Anthropic) client"""
        try:
            import anthropic
            
            claude_key = getattr(self.settings, 'ANTHROPIC_API_KEY', None)
            if claude_key:
                self.anthropic_client = anthropic.Anthropic(api_key=claude_key)
                
                # Test the connection
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
                
                self.available_models.append("claude")
                logger.info("Claude initialized successfully")
        except Exception as e:
            logger.warning(f"Claude initialization failed: {e}")

    async def detect_use_case(self, query: str, sector: str, model: str = None) -> str:
        """Intelligently detect the best use case for a query"""
        if self.demo_mode:
            return f"{sector} Analysis" if sector != "General" else "Quick Playbook Answers"
        
        use_cases = [
            "Quick Playbook Answers", "Lessons Learned", "Project Review / MOT",
            "TRL / RIRL Mapping", "Project Similarity", "Change Management", "Product Acceptance"
        ]
        
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
        """
        
        try:
            response = await self._make_ai_request(prompt, model or self.current_model, max_tokens=50)
            detected = response.strip()
            return detected if detected in use_cases else "Quick Playbook Answers"
        except Exception as e:
            logger.error(f"Error detecting use case: {e}")
            return "Quick Playbook Answers"

    async def generate_response(
        self,
        query: str,
        sector: str = "General",
        use_case: Optional[str] = None,
        user_type: str = "public",
        model: str = None
    ) -> Dict[str, Any]:
        """Generate AI response for chat"""
        
        if self.demo_mode:
            return {
                "response": f"[DEMO MODE] I received your message about '{query[:50]}...' in the {sector} sector. To enable real AI responses, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.",
                "confidence": 0.5,
                "sources": [{"title": "Demo Mode Active", "relevance": 1.0}],
                "suggested_use_case": use_case or f"{sector} Analysis" if sector != "General" else None,
                "model_used": "demo"
            }
        
        try:
            # Detect use case if not provided
            if not use_case:
                use_case = await self.detect_use_case(query, sector, model)
            
            # Create specialized prompt
            prompt = self._create_sector_prompt(query, sector, use_case, user_type)
            
            # Generate response with selected model
            selected_model = model or self.current_model
            ai_response = await self._make_ai_request(prompt, selected_model, max_tokens=800)
            
            # Calculate confidence
            confidence = self._calculate_response_confidence(ai_response, query)
            
            return {
                "response": ai_response,
                "confidence": confidence,
                "sources": self._generate_mock_sources(sector, use_case),
                "suggested_use_case": use_case,
                "model_used": selected_model
            }
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return {
                "response": f"I apologize, but I'm experiencing technical difficulties. Please try again later. (Error: {str(e)})",
                "confidence": 0.0,
                "sources": [],
                "suggested_use_case": use_case,
                "model_used": model or "error"
            }

    async def _make_ai_request(self, prompt: str, model: str, max_tokens: int = 800) -> str:
        """Make AI request to specified model"""
        
        if model == "openai":
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif model == "claude":
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        else:
            raise ValueError(f"Unsupported model: {model}")

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
        
        return f"""
        {base_context}
        
        {sector_context}
        
        User Question: {query}
        
        Please provide a detailed, helpful response that addresses their specific needs.
        """

    def _calculate_response_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score based on response quality"""
        confidence = 0.8
        
        if len(response) < 100:
            confidence -= 0.2
        elif len(response) > 500:
            confidence += 0.1
        
        quality_indicators = [
            "strategy", "framework", "approach", "recommend", "suggest",
            "analysis", "assessment", "evaluation", "implementation",
            "best practice", "guidance", "methodology"
        ]
        
        found_indicators = sum(1 for indicator in quality_indicators 
                             if indicator.lower() in response.lower())
        confidence += min(found_indicators * 0.02, 0.1)
        
        return min(confidence, 0.95)

    def _generate_mock_sources(self, sector: str, use_case: str) -> List[Dict[str, Any]]:
        """Generate realistic mock sources"""
        if sector.lower() == "transport":
            return [
                {"title": f"{sector} Strategy Framework 2024", "relevance": 0.92},
                {"title": f"Connected Places Catapult {use_case} Guide", "relevance": 0.88},
                {"title": f"UK {sector} Innovation Roadmap", "relevance": 0.85}
            ]
        else:
            return [
                {"title": f"{sector} Policy Guidelines", "relevance": 0.90},
                {"title": f"Strategic {use_case} Framework", "relevance": 0.87},
                {"title": f"{sector} Best Practices Compendium", "relevance": 0.84}
            ]

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models if not self.demo_mode else ["demo"]

    def set_model(self, model: str) -> bool:
        """Set the current model"""
        if model in self.available_models:
            self.current_model = model
            return True
        return False

# Global AI service instance
ai_service = AIService() 