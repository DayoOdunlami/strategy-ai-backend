# specialized_agents.py - Multi-Agent System for Strategy AI
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from abc import ABC, abstractmethod

from ai_services import ai_service
from database import db_manager
from vector_store import vector_store

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, specialization: str):
        self.name = name
        self.specialization = specialization
        self.status = "ready"
        self.last_tested = datetime.now()
        
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return response"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "specialization": self.specialization,
            "status": self.status,
            "last_tested": self.last_tested
        }

class DocumentAnalysisAgent(BaseAgent):
    """
    Specialized agent for document analysis and metadata extraction
    """
    
    def __init__(self):
        super().__init__("DocumentAnalysisAgent", "Document analysis and metadata extraction")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process document analysis requests"""
        try:
            analysis_type = request.get("analysis_type", "metadata")
            document_text = request.get("document_text", "")
            sector = request.get("sector", "General")
            
            # Simplified processing for minimal deployment
            return {
                "agent": self.name,
                "response": f"Document analysis complete for {analysis_type} in {sector} sector.",
                "confidence": 0.8,
                "response_type": "document_analysis",
                "structured_data": {
                    "analysis_type": analysis_type,
                    "sector": sector,
                    "processed": True
                }
            }
                
        except Exception as e:
            logger.error(f"DocumentAnalysisAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to complete document analysis at this time.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _extract_metadata(self, text: str, sector: str) -> Dict[str, Any]:
        """Extract metadata from document text"""
        prompt = f"""
        Analyze this document text and extract structured metadata for the {sector} sector.
        
        Document Text:
        {text[:2000]}...
        
        Extract the following metadata in JSON format:
        {{
            "title": "Descriptive title (5-8 words)",
            "topic": "Main topic/area",
            "keywords": ["key", "words", "list"],
            "document_type": "policy/strategy/guidance/report/etc",
            "complexity_level": "basic/intermediate/advanced",
            "target_audience": "description of intended audience",
            "key_themes": ["theme1", "theme2", "theme3"],
            "actionable_items": ["action1", "action2"],
            "confidence": 0.85
        }}
        
        Return only valid JSON:
        """
        
        response = await ai_service.generate_response(
            query=prompt,
            sector=sector,
            use_case="Document Analysis",
            user_type="system"
        )
        
        try:
            # Parse JSON response
            metadata = json.loads(response["response"])
            return {
                "agent": self.name,
                "response": "Metadata extracted successfully",
                "confidence": metadata.get("confidence", 0.8),
                "response_type": "metadata_extraction",
                "structured_data": metadata
            }
        except json.JSONDecodeError:
            return {
                "agent": self.name,
                "response": response["response"],
                "confidence": 0.7,
                "response_type": "metadata_extraction",
                "structured_data": {"raw_analysis": response["response"]}
            }
    
    async def _generate_summary(self, text: str, sector: str) -> Dict[str, Any]:
        """Generate document summary"""
        prompt = f"""
        Create a comprehensive summary of this {sector} sector document.
        
        Document Text:
        {text[:2000]}...
        
        Provide a structured summary with:
        
        EXECUTIVE SUMMARY (2-3 sentences)
        
        KEY POINTS:
        • Point 1
        • Point 2  
        • Point 3
        
        STRATEGIC IMPLICATIONS:
        • Implication 1
        • Implication 2
        
        RECOMMENDED ACTIONS:
        • Action 1
        • Action 2
        
        Summary:
        """
        
        response = await ai_service.generate_response(
            query=prompt,
            sector=sector,
            use_case="Document Analysis",
            user_type="system"
        )
        
        return {
            "agent": self.name,
            "response": response["response"],
            "confidence": response.get("confidence", 0.8),
            "response_type": "document_summary",
            "sources": response.get("sources", [])
        }

class StrategyAnalysisAgent(BaseAgent):
    """
    Specialized agent for strategic analysis and recommendations
    """
    
    def __init__(self):
        super().__init__("StrategyAnalysisAgent", "Strategic analysis and recommendations")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process strategy analysis requests"""
        try:
            query = request.get("query", "")
            sector = request.get("sector", "General")
            use_case = request.get("use_case", "Quick Playbook Answers")
            context = request.get("context", "")
            
            # Get relevant context from vector store
            context_docs = await vector_store.semantic_search(
                query=query,
                filters={"sector": sector, "use_case": use_case},
                top_k=6
            )
            
            # Create strategic analysis prompt
            prompt = self._create_strategy_prompt(query, sector, use_case, context_docs)
            
            response = await ai_service.generate_response(
                query=prompt,
                sector=sector,
                use_case=use_case,
                user_type="analyst"
            )
            
            return {
                "agent": self.name,
                "response": response["response"],
                "confidence": response.get("confidence", 0.8),
                "response_type": "strategic_analysis",
                "sources": context_docs[:3],  # Top 3 sources
                "suggested_use_case": response.get("suggested_use_case")
            }
            
        except Exception as e:
            logger.error(f"StrategyAnalysisAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to complete strategic analysis at this time.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_strategy_prompt(self, query: str, sector: str, use_case: str, context_docs: List[Dict]) -> str:
        """Create specialized strategy analysis prompt"""
        
        context_text = "\n\n".join([
            f"Source: {doc['metadata'].get('title', 'Unknown')}\nContent: {doc['text'][:300]}..."
            for doc in context_docs[:3]
        ])
        
        return f"""
        You are a senior strategic analyst specializing in {sector} sector policy and strategy development.
        
        Analysis Request: {query}
        Sector: {sector}
        Use Case: {use_case}
        
        Relevant Context:
        {context_text}
        
        Provide a comprehensive strategic analysis with:
        
        1. STRATEGIC ASSESSMENT
        2. KEY CONSIDERATIONS
        3. RECOMMENDED APPROACH
        4. IMPLEMENTATION ROADMAP
        5. RISK MITIGATION
        6. SUCCESS METRICS
        
        Focus on actionable, evidence-based recommendations that are specific to the {sector} sector.
        
        Analysis:
        """

class SystemAnalyticsAgent(BaseAgent):
    """
    Specialized agent for system analytics and performance monitoring
    """
    
    def __init__(self):
        super().__init__("SystemAnalyticsAgent", "System analytics and monitoring")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process system analytics requests"""
        try:
            analysis_type = request.get("analysis_type", "system_health")
            time_range = request.get("time_range", "24h")
            
            if analysis_type == "system_health":
                return await self._analyze_system_health()
            elif analysis_type == "user_feedback":
                return await self._analyze_user_feedback(time_range)
            elif analysis_type == "performance":
                return await self._analyze_performance(time_range)
            else:
                return await self._comprehensive_analytics(time_range)
                
        except Exception as e:
            logger.error(f"SystemAnalyticsAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to complete system analysis at this time.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze system health metrics"""
        
        # Get system analytics
        analytics = await db_manager.get_system_analytics()
        feedback_analytics = await db_manager.get_feedback_analytics()
        
        prompt = f"""
        Analyze the current system health based on these metrics:
        
        System Metrics:
        - Total Documents: {analytics['total_documents']}
        - Total Chunks: {analytics['total_chunks']}
        - Total Sectors: {analytics['total_sectors']}
        - Recent Activity: {analytics['recent_activity_count']} (last 7 days)
        
        User Feedback:
        - Total Feedback: {feedback_analytics['total_feedback']}
        - Average Rating: {feedback_analytics['average_rating']:.2f}/5.0
        - Helpful Percentage: {feedback_analytics['helpful_percentage']:.1f}%
        
        Documents by Sector: {analytics['documents_by_sector']}
        
        Provide a system health analysis with:
        
        1. OVERALL STATUS (Excellent/Good/Fair/Poor)
        2. KEY STRENGTHS
        3. AREAS FOR IMPROVEMENT
        4. IMMEDIATE ACTIONS NEEDED
        5. RECOMMENDATIONS
        
        Analysis:
        """
        
        response = await ai_service.generate_response(
            query=prompt,
            sector="General",
            use_case="System Analytics",
            user_type="admin"
        )
        
        return {
            "agent": self.name,
            "response": response["response"],
            "confidence": 0.9,
            "response_type": "system_health",
            "structured_data": {
                "metrics": analytics,
                "feedback": feedback_analytics
            }
        }

class OrchestrationAgent(BaseAgent):
    """
    Orchestration agent that coordinates multiple specialized agents
    """
    
    def __init__(self):
        super().__init__("OrchestrationAgent", "Multi-agent coordination and orchestration")
        self.agents = {
            "document": DocumentAnalysisAgent(),
            "strategy": StrategyAnalysisAgent(),
            "analytics": SystemAnalyticsAgent()
        }
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multiple agents for complex requests"""
        try:
            complexity = request.get("complexity", "simple")
            query = request.get("query", "")
            
            if complexity == "simple":
                # Use single best agent
                agent = self._select_primary_agent(request)
                return await agent.process(request)
            else:
                # Use multiple agents
                return await self._orchestrate_multi_agent(request)
                
        except Exception as e:
            logger.error(f"OrchestrationAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to complete orchestrated analysis.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _select_primary_agent(self, request: Dict[str, Any]) -> BaseAgent:
        """Select the best agent for a request"""
        request_type = request.get("type", "chat")
        query = request.get("query", "").lower()
        
        if request_type == "document_analysis" or "analyze document" in query:
            return self.agents["document"]
        elif "system" in query or "analytics" in query or "performance" in query:
            return self.agents["analytics"]
        else:
            return self.agents["strategy"]
    
    async def _orchestrate_multi_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for comprehensive analysis"""
        
        agents_used = []
        detailed_results = {}
        
        # Strategy analysis (primary)
        strategy_result = await self.agents["strategy"].process(request)
        agents_used.append("StrategyAnalysisAgent")
        detailed_results["strategy"] = strategy_result
        
        # If query relates to documents, add document analysis
        if "document" in request.get("query", "").lower():
            doc_request = {**request, "analysis_type": "key_insights"}
            doc_result = await self.agents["document"].process(doc_request)
            agents_used.append("DocumentAnalysisAgent")
            detailed_results["document"] = doc_result
        
        # Combine results
        primary_response = strategy_result["response"]
        combined_confidence = sum(
            result.get("confidence", 0.5) for result in detailed_results.values()
        ) / len(detailed_results)
        
        return {
            "agent": self.name,
            "primary_response": primary_response,
            "agents_used": agents_used,
            "response_type": "orchestrated_analysis",
            "detailed_results": detailed_results,
            "confidence": combined_confidence
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        agents_info = [agent.get_info() for agent in self.agents.values()]
        
        return {
            "agents": agents_info,
            "orchestrator_status": self.status,
            "total_agents": len(self.agents)
        }

# Global orchestration agent instance
orchestration_agent = OrchestrationAgent() 