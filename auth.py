# auth.py - Authentication System
from typing import Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Header, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
from config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

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

# JWT Authentication Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    return {"sub": username}

# Global admin auth instance
admin_auth = AdminAuth()