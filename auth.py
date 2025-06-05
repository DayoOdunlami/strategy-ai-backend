# auth.py - Simple Admin Authentication
from typing import Optional
from fastapi import HTTPException, Header, Depends
import logging
from config import settings

logger = logging.getLogger(__name__)

class AdminAuth:
    """
    Simple admin authentication for API endpoints
    Uses API key-based authentication for simplicity
    """
    
    def __init__(self):
        self.admin_key = settings.ADMIN_API_KEY
        if not self.admin_key or self.admin_key == "admin-secret-key-change-in-production":
            logger.warning("Using default admin API key - change this in production!")

    async def verify_admin(self, admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> str:
        """
        Verify admin API key from header
        Used as FastAPI dependency for admin endpoints
        """
        if not admin_key:
            logger.warning("Admin endpoint accessed without API key")
            raise HTTPException(
                status_code=401,
                detail="Admin API key required. Provide X-Admin-Key header.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if admin_key != self.admin_key:
            logger.warning(f"Invalid admin API key attempted: {admin_key[:10]}...")
            raise HTTPException(
                status_code=401,
                detail="Invalid admin API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        logger.info("Admin authentication successful")
        return admin_key

    async def verify_admin_query(self, admin_key: Optional[str] = None) -> str:
        """
        Alternative verification method using query parameter
        For endpoints that need query-based auth
        """
        if not admin_key:
            raise HTTPException(
                status_code=401,
                detail="Admin API key required as query parameter 'admin_key'"
            )
        
        if admin_key != self.admin_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid admin API key"
            )
        
        return admin_key

    def is_valid_admin_key(self, admin_key: str) -> bool:
        """
        Simple validation without raising exceptions
        Useful for conditional logic
        """
        return admin_key == self.admin_key

    def get_auth_info(self) -> dict:
        """Get authentication configuration info"""
        return {
            "auth_method": "API Key",
            "header_name": "X-Admin-Key",
            "key_configured": bool(self.admin_key),
            "using_default_key": self.admin_key == "admin-secret-key-change-in-production"
        }