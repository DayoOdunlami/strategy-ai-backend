from fastapi import WebSocket
from typing import Dict, Set, Optional
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Custom error class for processing errors"""
    def __init__(self, message: str, status_code: int, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

class WebSocketManager:
    def __init__(self):
        # Store active connections by document ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Track last update time for rate limiting
        self.last_update: Dict[str, datetime] = {}
        # Track connection attempts for rate limiting
        self.connection_attempts: Dict[str, int] = {}
        # Minimum time between updates
        self.min_update_interval = timedelta(milliseconds=100)
        # Maximum connection attempts
        self.max_connection_attempts = 5
        # Store processing status
        self.processing_status: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, document_id: str):
        """Connect a new WebSocket client with rate limiting"""
        # Check connection attempts
        current_time = datetime.now()
        if document_id in self.connection_attempts:
            if self.connection_attempts[document_id] >= self.max_connection_attempts:
                last_attempt = self.last_update.get(document_id, datetime.min)
                if current_time - last_attempt < timedelta(minutes=5):
                    raise ProcessingError(
                        message="Too many connection attempts",
                        status_code=429,
                        details={"retry_after": "5 minutes"}
                    )
                else:
                    # Reset counter after 5 minutes
                    self.connection_attempts[document_id] = 0
            self.connection_attempts[document_id] += 1
        else:
            self.connection_attempts[document_id] = 1

        await websocket.accept()
        
        if document_id not in self.active_connections:
            self.active_connections[document_id] = set()
        self.active_connections[document_id].add(websocket)
        
        # Send current status if exists
        if document_id in self.processing_status:
            await websocket.send_json(self.processing_status[document_id])
        
        logger.info(f"New WebSocket connection for document {document_id}")

    def disconnect(self, websocket: WebSocket, document_id: str):
        """Disconnect a WebSocket client"""
        if document_id in self.active_connections:
            self.active_connections[document_id].discard(websocket)
            if not self.active_connections[document_id]:
                del self.active_connections[document_id]
                # Clean up tracking data
                self.last_update.pop(document_id, None)
                self.connection_attempts.pop(document_id, None)
        
        logger.info(f"WebSocket disconnected for document {document_id}")

    async def broadcast_update(self, document_id: str, message: dict):
        """Broadcast an update to all clients for a document with rate limiting"""
        current_time = datetime.now()
        
        # Check rate limit
        if document_id in self.last_update:
            time_since_last = current_time - self.last_update[document_id]
            if time_since_last < self.min_update_interval:
                return  # Skip update if too soon
        
        if document_id in self.active_connections:
            dead_connections = set()
            
            for connection in self.active_connections[document_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                self.active_connections[document_id].discard(dead)
            
            if not self.active_connections[document_id]:
                del self.active_connections[document_id]
            else:
                # Update tracking only for successful broadcasts
                self.last_update[document_id] = current_time
                # Store latest status
                if message.get("type") == "processing_update":
                    self.processing_status[document_id] = message

    async def send_processing_update(
        self,
        document_id: str,
        status: str,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        details: Optional[dict] = None
    ):
        """Send a document processing status update"""
        update = {
            "type": "processing_update",
            "document_id": document_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if progress is not None:
            update["progress"] = progress
        
        if message:
            update["message"] = message
            
        if details:
            update["details"] = details
            
        await self.broadcast_update(document_id, update)

    async def send_error(self, document_id: str, error: str, status_code: int = 500, details: dict = None):
        """Send an enhanced error message"""
        error_message = {
            "type": "error",
            "document_id": document_id,
            "error": error,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            error_message["details"] = details
            
        await self.broadcast_update(document_id, error_message)
        # Clear status on error
        self.processing_status.pop(document_id, None)

    def get_active_processes(self) -> Dict[str, dict]:
        """Get all active processing statuses"""
        return {
            doc_id: status
            for doc_id, status in self.processing_status.items()
            if doc_id in self.active_connections
        }

    async def cleanup_stale_connections(self, max_age_minutes: int = 30):
        """Clean up stale connections and status data"""
        current_time = datetime.now()
        stale_threshold = current_time - timedelta(minutes=max_age_minutes)
        
        stale_docs = [
            doc_id
            for doc_id, last_update in self.last_update.items()
            if last_update < stale_threshold
        ]
        
        for doc_id in stale_docs:
            if doc_id in self.active_connections:
                connections = self.active_connections[doc_id].copy()
                for connection in connections:
                    try:
                        await connection.close(code=1000, reason="Connection timed out")
                    except Exception as e:
                        logger.error(f"Error closing stale connection: {e}")
                    self.disconnect(connection, doc_id)
            
            # Clean up all tracking data
            self.last_update.pop(doc_id, None)
            self.connection_attempts.pop(doc_id, None)
            self.processing_status.pop(doc_id, None) 