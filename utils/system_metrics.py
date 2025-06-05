"""
Utility functions for collecting system metrics and feedback data
"""
import psutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from models import SystemMetric, PerformanceMetric

async def collect_system_metrics() -> List[SystemMetric]:
    """Collect current system metrics"""
    metrics = []
    
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    metrics.append(SystemMetric(
        name="cpu_usage",
        value=cpu_percent,
        unit="percent",
        timestamp=datetime.now(),
        category="cpu",
        status="warning" if cpu_percent > 80 else "normal"
    ))
    
    # Memory metrics
    memory = psutil.virtual_memory()
    metrics.append(SystemMetric(
        name="memory_usage",
        value=memory.percent,
        unit="percent",
        timestamp=datetime.now(),
        category="memory",
        status="warning" if memory.percent > 85 else "normal"
    ))
    
    # Disk metrics
    disk = psutil.disk_usage('/')
    metrics.append(SystemMetric(
        name="disk_usage",
        value=disk.percent,
        unit="percent",
        timestamp=datetime.now(),
        category="disk",
        status="warning" if disk.percent > 90 else "normal"
    ))
    
    return metrics

async def collect_performance_metrics(time_range: str) -> List[PerformanceMetric]:
    """Collect performance metrics over time range"""
    # Convert time range to timedelta
    if time_range.endswith('h'):
        delta = timedelta(hours=int(time_range[:-1]))
    elif time_range.endswith('d'):
        delta = timedelta(days=int(time_range[:-1]))
    else:
        delta = timedelta(hours=24)  # default to 24h
    
    start_time = datetime.now() - delta
    
    # TODO: Implement actual metrics collection from your monitoring system
    # This is a placeholder that returns sample metrics
    return [
        PerformanceMetric(
            endpoint="/api/chat",
            response_time=0.245,
            error_rate=0.01,
            requests_per_minute=12.5,
            timestamp=datetime.now(),
            status="normal"
        ),
        PerformanceMetric(
            endpoint="/api/documents",
            response_time=1.123,
            error_rate=0.02,
            requests_per_minute=5.7,
            timestamp=datetime.now(),
            status="normal"
        )
    ]

async def collect_user_feedback(time_range: str) -> List[Dict[str, Any]]:
    """Collect user feedback data over time range"""
    # Convert time range to timedelta
    if time_range.endswith('h'):
        delta = timedelta(hours=int(time_range[:-1]))
    elif time_range.endswith('d'):
        delta = timedelta(days=int(time_range[:-1]))
    else:
        delta = timedelta(days=7)  # default to 7d
    
    start_time = datetime.now() - delta
    
    # TODO: Implement actual feedback collection from your database
    # This is a placeholder that returns sample feedback
    return [
        {
            "type": "rating",
            "value": 4.5,
            "timestamp": datetime.now().isoformat(),
            "category": "response_quality"
        },
        {
            "type": "comment",
            "value": "Very helpful responses, but could be faster",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "category": "performance"
        }
    ]

async def analyze_system_health(metrics: List[SystemMetric]) -> Dict[str, Any]:
    """Analyze system health metrics"""
    issues = []
    recommendations = []
    health_score = 100.0
    
    for metric in metrics:
        if metric.status == "warning":
            health_score -= 10
            issues.append({
                "metric": metric.name,
                "value": metric.value,
                "threshold": "80%" if metric.name == "cpu_usage" else "85%" if metric.name == "memory_usage" else "90%",
                "category": metric.category
            })
            recommendations.append(f"Investigate high {metric.name} ({metric.value}%)")
    
    return {
        "status": "degraded" if issues else "healthy",
        "health_score": max(0, health_score),
        "issues": issues,
        "recommendations": recommendations
    }

async def analyze_performance(metrics: List[PerformanceMetric]) -> Dict[str, Any]:
    """Analyze performance metrics"""
    bottlenecks = []
    optimizations = []
    performance_score = 100.0
    
    for metric in metrics:
        if metric.response_time > 1.0:
            performance_score -= 10
            bottlenecks.append({
                "endpoint": metric.endpoint,
                "response_time": metric.response_time,
                "threshold": 1.0
            })
            optimizations.append(f"Optimize response time for {metric.endpoint}")
        
        if metric.error_rate > 0.01:
            performance_score -= 15
            bottlenecks.append({
                "endpoint": metric.endpoint,
                "error_rate": metric.error_rate,
                "threshold": 0.01
            })
            optimizations.append(f"Investigate errors in {metric.endpoint}")
    
    return {
        "overall_status": "degraded" if bottlenecks else "healthy",
        "performance_score": max(0, performance_score),
        "bottlenecks": bottlenecks,
        "optimizations": optimizations
    }

async def analyze_feedback(feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze user feedback data"""
    if not feedback_data:
        return {
            "satisfaction_score": 0.0,
            "feedback_count": 0,
            "sentiment_analysis": {},
            "key_themes": [],
            "recommendations": ["Start collecting user feedback"]
        }
    
    # Calculate satisfaction score from ratings
    ratings = [item["value"] for item in feedback_data if item["type"] == "rating"]
    satisfaction_score = sum(ratings) / len(ratings) if ratings else 0.0
    
    # Analyze sentiment and themes from comments
    comments = [item for item in feedback_data if item["type"] == "comment"]
    # TODO: Implement actual sentiment analysis
    sentiment = {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
    
    # Extract key themes
    # TODO: Implement actual theme extraction
    themes = [
        {"theme": "Response Quality", "count": 5},
        {"theme": "Performance", "count": 3}
    ]
    
    return {
        "satisfaction_score": satisfaction_score,
        "feedback_count": len(feedback_data),
        "sentiment_analysis": sentiment,
        "key_themes": themes,
        "recommendations": [
            "Focus on areas with low satisfaction",
            "Address common themes in negative feedback"
        ]
    } 