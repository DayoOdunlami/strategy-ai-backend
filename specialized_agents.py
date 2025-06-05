class SystemAnalyticsAgent(BaseAgent):
    """
    Specialized agent for system analytics, monitoring, and user feedback
    Handles system health, performance metrics, and user experience analysis
    """
    
    def __init__(self):
        super().__init__("SystemAnalyticsAgent", "System analytics and monitoring")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process system analytics requests"""
        try:
            analysis_type = request.get("analysis_type", "system_health")
            time_range = request.get("time_range", "24h")
            metrics = request.get("metrics", [])
            feedback_data = request.get("feedback_data", [])
            
            prompt = self._create_analytics_prompt(analysis_type, time_range, metrics, feedback_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            analysis_content = response.choices[0].message.content.strip()
            structured_analysis = self._structure_analytics_response(analysis_content, analysis_type)
            
            return {
                "agent": self.name,
                "response": analysis_content,
                "structured_analysis": structured_analysis,
                "confidence": 0.9,
                "response_type": "system_analytics",
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            logger.error(f"SystemAnalyticsAgent error: {e}")
            return {
                "agent": self.name,
                "response": "Unable to complete system analysis at this time.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_analytics_prompt(
        self, 
        analysis_type: str, 
        time_range: str, 
        metrics: List[Dict[str, Any]], 
        feedback_data: List[Dict[str, Any]]
    ) -> str:
        """Create prompt based on analysis type"""
        
        if analysis_type == "system_health":
            return f"""
            Analyze system health metrics over the past {time_range}.
            
            Metrics Data:
            {json.dumps(metrics, indent=2)}
            
            Provide a comprehensive system health analysis with:
            
            1. SYSTEM STATUS SUMMARY
            2. KEY METRICS ANALYSIS
            3. PERFORMANCE TRENDS
            4. POTENTIAL ISSUES
            5. RECOMMENDATIONS
            6. PRIORITY ACTIONS
            
            Analysis:
            """
            
        elif analysis_type == "user_feedback":
            return f"""
            Analyze user feedback and satisfaction metrics over {time_range}.
            
            Feedback Data:
            {json.dumps(feedback_data, indent=2)}
            
            Provide a detailed user feedback analysis with:
            
            1. FEEDBACK OVERVIEW
            2. SATISFACTION METRICS
            3. KEY THEMES
            4. PAIN POINTS
            5. POSITIVE HIGHLIGHTS
            6. ACTIONABLE INSIGHTS
            7. IMPROVEMENT RECOMMENDATIONS
            
            Analysis:
            """
            
        elif analysis_type == "performance_metrics":
            return f"""
            Analyze system performance metrics over {time_range}.
            
            Performance Data:
            {json.dumps(metrics, indent=2)}
            
            Provide a detailed performance analysis with:
            
            1. PERFORMANCE SUMMARY
            2. RESPONSE TIME ANALYSIS
            3. ERROR RATE ANALYSIS
            4. RESOURCE UTILIZATION
            5. BOTTLENECKS
            6. OPTIMIZATION RECOMMENDATIONS
            
            Analysis:
            """
            
        else:
            return f"""
            Provide a general system analysis over {time_range}.
            
            Available Data:
            Metrics: {json.dumps(metrics, indent=2)}
            Feedback: {json.dumps(feedback_data, indent=2)}
            
            Generate a comprehensive analysis with:
            
            1. OVERVIEW
            2. KEY FINDINGS
            3. METRICS ANALYSIS
            4. RECOMMENDATIONS
            
            Analysis:
            """
    
    def _structure_analytics_response(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Structure the analytics response based on type"""
        sections = {}
        current_section = None
        current_content = []
        
        # Parse sections from content
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.isupper() or (line[0].isdigit() and '.' in line[:5]):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.lower().replace('.', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # Add metadata
        analysis_metadata = {
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "sections_found": list(sections.keys())
        }
        
        return {
            "sections": sections,
            "metadata": analysis_metadata
        } 