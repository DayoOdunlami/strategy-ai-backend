# specialized_agents.py - Specialized AI Agents System
import asyncio
from typing import Dict, List, Optional, Any, Union
import logging
import openai
from datetime import datetime
import json
from config import settings

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all specialized AI agents"""
    
    def __init__(self, name: str, specialization: str):
        self.name = name
        self.specialization = specialization
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.AI_MODEL_NAME
        self.temperature = settings.AI_TEMPERATURE
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Base processing method - to be overridden by specialized agents"""
        raise NotImplementedError("Each agent must implement its own process method")

class ChatAgent(BaseAgent):
    """
    Specialized agent for conversational responses
    Handles quick answers, clarifications, and general chat
    """
    
    def __init__(self):
        super().__init__("ChatAgent", "Conversational responses and quick answers")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversational responses"""
        try:
            query = request.get("query", "")
            context = request.get("context", "")
            user_type = request.get("user_type", "public")
            use_case = request.get("use_case", "Quick Playbook Answers")
            
            prompt = self._create_chat_prompt(query, context, user_type, use_case)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            return {
                "agent": self.name,
                "response": response.choices[0].message.content.strip(),
                "confidence": 0.85,
                "response_type": "conversational"
            }
            
        except Exception as e:
            logger.error(f"ChatAgent error: {e}")
            return {
                "agent": self.name,
                "response": "I'm having trouble processing your request right now. Please try again.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_chat_prompt(self, query: str, context: str, user_type: str, use_case: str) -> str:
        return f"""
        You are a conversational AI assistant specializing in strategic planning and project management.
        
        User Type: {user_type}
        Use Case: {use_case}
        
        Provide a direct, conversational response to the user's question based on the context provided.
        Be helpful, clear, and actionable. Reference specific information from the context when relevant.
        
        Context Documents:
        {context}
        
        User Question: {query}
        
        Response (conversational and helpful):
        """

class AnalysisAgent(BaseAgent):
    """
    Specialized agent for strategic analysis and deep thinking
    Handles complex analysis, insights, and strategic recommendations
    """
    
    def __init__(self):
        super().__init__("AnalysisAgent", "Strategic analysis and insights")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic analysis and insights"""
        try:
            query = request.get("query", "")
            context = request.get("context", "")
            analysis_type = request.get("analysis_type", "general")
            sector = request.get("sector", "General")
            
            prompt = self._create_analysis_prompt(query, context, analysis_type, sector)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            
            # Parse structured analysis
            analysis_content = response.choices[0].message.content.strip()
            structured_analysis = self._parse_analysis_response(analysis_content)
            
            return {
                "agent": self.name,
                "response": analysis_content,
                "structured_analysis": structured_analysis,
                "confidence": 0.9,
                "response_type": "strategic_analysis"
            }
            
        except Exception as e:
            logger.error(f"AnalysisAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to complete strategic analysis at this time.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_analysis_prompt(self, query: str, context: str, analysis_type: str, sector: str) -> str:
        return f"""
        You are a strategic analysis AI specializing in {sector} sector planning and project management.
        
        Perform a comprehensive strategic analysis of the question using the provided context.
        
        Analysis Type: {analysis_type}
        Sector Focus: {sector}
        
        Structure your analysis as follows:
        
        EXECUTIVE SUMMARY:
        [2-3 sentence overview of key findings]
        
        KEY INSIGHTS:
        [3-5 bullet points of main insights]
        
        STRATEGIC IMPLICATIONS:
        [Analysis of what this means strategically]
        
        RECOMMENDATIONS:
        [Specific, actionable recommendations]
        
        RISK FACTORS:
        [Potential risks or challenges to consider]
        
        Context Documents:
        {context}
        
        Analysis Question: {query}
        
        Strategic Analysis:
        """
    
    def _parse_analysis_response(self, content: str) -> Dict[str, List[str]]:
        """Parse structured analysis response into components"""
        structured = {
            "executive_summary": [],
            "key_insights": [],
            "strategic_implications": [],
            "recommendations": [],
            "risk_factors": []
        }
        
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if "EXECUTIVE SUMMARY" in line.upper():
                current_section = "executive_summary"
            elif "KEY INSIGHTS" in line.upper():
                current_section = "key_insights"
            elif "STRATEGIC IMPLICATIONS" in line.upper():
                current_section = "strategic_implications"
            elif "RECOMMENDATIONS" in line.upper():
                current_section = "recommendations"
            elif "RISK FACTORS" in line.upper():
                current_section = "risk_factors"
            elif current_section and line.startswith(('•', '-', '*', '1.', '2.')):
                # Extract bullet points
                clean_line = line.lstrip('•-*123456789. ').strip()
                if clean_line:
                    structured[current_section].append(clean_line)
            elif current_section == "executive_summary" and not line.startswith(('•', '-', '*')):
                # Executive summary might not be bulleted
                structured[current_section].append(line)
        
        return structured

class ReportAgent(BaseAgent):
    """
    Specialized agent for generating comprehensive reports
    Handles document creation, formatting, and structured reporting
    """
    
    def __init__(self):
        super().__init__("ReportAgent", "Report generation and document creation")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive reports"""
        try:
            report_type = request.get("report_type", "strategy_analysis")
            context = request.get("context", "")
            parameters = request.get("parameters", {})
            sector = request.get("sector", "General")
            
            prompt = self._create_report_prompt(report_type, context, parameters, sector)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            report_content = response.choices[0].message.content.strip()
            
            # Structure the report
            structured_report = self._structure_report(report_content, report_type)
            
            return {
                "agent": self.name,
                "response": report_content,
                "structured_report": structured_report,
                "report_type": report_type,
                "confidence": 0.95,
                "response_type": "comprehensive_report"
            }
            
        except Exception as e:
            logger.error(f"ReportAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to generate report at this time.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_report_prompt(self, report_type: str, context: str, parameters: Dict, sector: str) -> str:
        if report_type == "strategy_analysis":
            return f"""
            Generate a comprehensive strategic analysis report for the {sector} sector.
            
            Create a professional report with the following structure:
            
            1. EXECUTIVE SUMMARY
            2. SITUATION ANALYSIS
            3. KEY FINDINGS
            4. STRATEGIC RECOMMENDATIONS
            5. IMPLEMENTATION ROADMAP
            6. RISK ASSESSMENT
            7. SUCCESS METRICS
            8. CONCLUSION
            
            Use the provided context to support your analysis with specific evidence and examples.
            Make recommendations actionable and sector-specific.
            
            Parameters: {json.dumps(parameters)}
            
            Context Documents:
            {context}
            
            Professional Strategic Analysis Report:
            """
        
        elif report_type == "project_similarity":
            return f"""
            Generate a comprehensive project similarity analysis report.
            
            Structure:
            1. ANALYSIS OVERVIEW
            2. SIMILAR PROJECTS IDENTIFIED
            3. COMPARATIVE ANALYSIS
            4. LESSONS LEARNED
            5. SUCCESS FACTORS
            6. RISK PATTERNS
            7. RECOMMENDATIONS FOR CURRENT PROJECT
            8. APPENDIX: PROJECT DETAILS
            
            Context: {context}
            Parameters: {json.dumps(parameters)}
            
            Project Similarity Analysis Report:
            """
        
        elif report_type == "lessons_learned":
            return f"""
            Generate a comprehensive lessons learned report.
            
            Structure:
            1. REPORT SUMMARY
            2. METHODOLOGY
            3. KEY LESSONS BY CATEGORY
            4. SUCCESS STORIES
            5. FAILURE ANALYSIS
            6. BEST PRACTICES
            7. RECOMMENDATIONS
            8. IMPLEMENTATION GUIDELINES
            
            Context: {context}
            Parameters: {json.dumps(parameters)}
            
            Lessons Learned Report:
            """
        
        else:
            return f"""
            Generate a comprehensive {report_type} report for the {sector} sector.
            
            Create a well-structured, professional report based on the context provided.
            Include executive summary, analysis, findings, and recommendations.
            
            Context: {context}
            Parameters: {json.dumps(parameters)}
            
            Report:
            """
    
    def _structure_report(self, content: str, report_type: str) -> Dict[str, Any]:
        """Structure report content into sections"""
        sections = {}
        current_section = "introduction"
        current_content = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header (numbered or all caps)
            if (line.upper() == line and len(line) > 5) or \
               (line.startswith(tuple('12345678')) and '.' in line and line.index('.') < 3):
                
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.lower().replace('.', '').replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return {
            "sections": sections,
            "word_count": len(content.split()),
            "generated_at": datetime.now().isoformat()
        }

class MetadataAgent(BaseAgent):
    """
    Specialized agent for content classification and metadata generation
    Handles tagging, categorization, and content analysis
    """
    
    def __init__(self):
        super().__init__("MetadataAgent", "Content classification and metadata generation")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed metadata for content"""
        try:
            content = request.get("content", "")
            sector = request.get("sector", "General")
            existing_metadata = request.get("existing_metadata", {})
            
            prompt = self._create_metadata_prompt(content, sector, existing_metadata)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600
            )
            
            metadata_response = response.choices[0].message.content.strip()
            structured_metadata = self._parse_metadata_response(metadata_response)
            
            return {
                "agent": self.name,
                "metadata": structured_metadata,
                "confidence": 0.9,
                "response_type": "metadata_analysis"
            }
            
        except Exception as e:
            logger.error(f"MetadataAgent error: {e}")
            return {
                "agent": self.name,
                "metadata": {"error": str(e)},
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_metadata_prompt(self, content: str, sector: str, existing_metadata: Dict) -> str:
        return f"""
        Analyze this content and generate comprehensive metadata for a {sector} sector document management system.
        
        Content to analyze:
        {content[:1500]}...
        
        Existing metadata (if any): {json.dumps(existing_metadata)}
        
        Generate the following metadata in this exact format:
        
        TITLE: [Descriptive document title]
        TOPIC: [Primary topic/subject area]
        CATEGORIES: [Primary category, Secondary category]
        TAGS: [keyword1, keyword2, keyword3, keyword4, keyword5]
        SUMMARY: [2-sentence summary]
        DOCUMENT_TYPE: [report, policy, guideline, analysis, etc.]
        COMPLEXITY_LEVEL: [basic, intermediate, advanced]
        TARGET_AUDIENCE: [who should read this]
        KEY_CONCEPTS: [main concept 1, main concept 2, main concept 3]
        RELATED_DOMAINS: [related area 1, related area 2]
        CONTENT_QUALITY: [high, medium, low]
        STRATEGIC_VALUE: [high, medium, low]
        
        Be specific and accurate. Focus on strategic and operational relevance.
        """
    
    def _parse_metadata_response(self, content: str) -> Dict[str, Any]:
        """Parse metadata response into structured format"""
        metadata = {}
        
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                clean_key = key.strip().lower().replace(' ', '_')
                clean_value = value.strip()
                
                # Parse list values
                if clean_key in ['categories', 'tags', 'key_concepts', 'related_domains']:
                    if clean_value:
                        metadata[clean_key] = [item.strip() for item in clean_value.split(',')]
                    else:
                        metadata[clean_key] = []
                else:
                    metadata[clean_key] = clean_value
        
        return metadata

class OrchestrationAgent:
    """
    Orchestration agent that coordinates specialized agents
    Determines which agents to use and combines their outputs
    """
    
    def __init__(self):
        self.chat_agent = ChatAgent()
        self.analysis_agent = AnalysisAgent()
        self.report_agent = ReportAgent()
        self.metadata_agent = MetadataAgent()
        
        logger.info("Initialized orchestration agent with all specialists")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration method - determines which agents to use
        and coordinates their responses
        """
        try:
            request_type = request.get("type", "chat")
            complexity = request.get("complexity", "simple")
            
            if request_type == "chat":
                return await self._handle_chat_request(request)
            elif request_type == "analysis":
                return await self._handle_analysis_request(request)
            elif request_type == "report":
                return await self._handle_report_request(request)
            elif request_type == "metadata":
                return await self._handle_metadata_request(request)
            elif request_type == "comprehensive":
                return await self._handle_comprehensive_request(request)
            else:
                # Default to chat
                return await self.chat_agent.process(request)
                
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return {
                "error": "Orchestration failed",
                "message": "Unable to process request with specialized agents",
                "fallback_available": True
            }
    
    async def _handle_chat_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simple chat requests"""
        return await self.chat_agent.process(request)
    
    async def _handle_analysis_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests requiring strategic analysis"""
        
        # For complex analysis, use both analysis and chat agents
        analysis_result = await self.analysis_agent.process(request)
        
        # If analysis was successful and request asks for conversational format
        if request.get("conversational_format", False) and analysis_result.get("confidence", 0) > 0.5:
            # Convert analysis to conversational format
            chat_request = {
                **request,
                "context": analysis_result["response"],
                "query": f"Explain this analysis in a conversational way: {request.get('query', '')}"
            }
            chat_result = await self.chat_agent.process(chat_request)
            
            return {
                "primary_response": chat_result["response"],
                "detailed_analysis": analysis_result,
                "agents_used": ["AnalysisAgent", "ChatAgent"],
                "response_type": "hybrid_analysis"
            }
        
        return analysis_result
    
    async def _handle_report_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation requests"""
        return await self.report_agent.process(request)
    
    async def _handle_metadata_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle metadata generation requests"""
        return await self.metadata_agent.process(request)
    
    async def _handle_comprehensive_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests that need multiple agents"""
        
        results = {}
        
        # Run multiple agents in parallel for comprehensive analysis
        tasks = []
        
        if request.get("include_analysis", True):
            tasks.append(("analysis", self.analysis_agent.process({**request, "type": "analysis"})))
        
        if request.get("include_metadata", False):
            tasks.append(("metadata", self.metadata_agent.process(request)))
        
        if request.get("generate_report", False):
            tasks.append(("report", self.report_agent.process({**request, "type": "report"})))
        
        # Execute tasks
        for task_name, task in tasks:
            try:
                results[task_name] = await task
            except Exception as e:
                logger.error(f"Error in {task_name} task: {e}")
                results[task_name] = {"error": str(e)}
        
        # Generate final conversational response combining all insights
        if results:
            summary_request = {
                **request,
                "context": self._combine_agent_outputs(results),
                "query": f"Summarize these comprehensive insights: {request.get('query', '')}"
            }
            final_response = await self.chat_agent.process(summary_request)
            
            return {
                "summary": final_response["response"],
                "detailed_results": results,
                "agents_used": list(results.keys()) + ["ChatAgent"],
                "response_type": "comprehensive_multi_agent"
            }
        
        # Fallback to simple chat
        return await self.chat_agent.process(request)
    
    def _combine_agent_outputs(self, results: Dict[str, Any]) -> str:
        """Combine outputs from multiple agents into context"""
        combined = []
        
        for agent_type, result in results.items():
            if "response" in result:
                combined.append(f"{agent_type.title()} Analysis:\n{result['response']}\n")
        
        return "\n".join(combined)
    
    def get_available_agents(self) -> List[Dict[str, str]]:
        """Get information about available agents"""
        return [
            {"name": "ChatAgent", "specialization": self.chat_agent.specialization},
            {"name": "AnalysisAgent", "specialization": self.analysis_agent.specialization},
            {"name": "ReportAgent", "specialization": self.report_agent.specialization},
            {"name": "MetadataAgent", "specialization": self.metadata_agent.specialization}
        ]