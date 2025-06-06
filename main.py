import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging
from datetime import datetime

# Import our custom modules
from config import Settings
from auth import get_current_user, create_access_token, verify_password, get_password_hash

# Initialize settings
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server Configuration
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))  # Properly handle Railway's PORT

app = FastAPI(
    title="Strategy AI Backend",
    description="Enhanced FastAPI backend for Strategy AI platform",
    version="2.0.1"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

@app.get("/")
async def root():
    return {
        "message": "Strategy AI Backend is running!",
        "version": "2.0.1",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "enhancement": "Authentication system enabled"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "local")
    }

@app.post("/auth/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """Simple login endpoint for testing authentication"""
    # For now, use hardcoded credentials (in production, use database)
    if login_request.username == "admin" and login_request.password == "password":
        access_token = create_access_token(data={"sub": login_request.username})
        return LoginResponse(access_token=access_token, token_type="bearer")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/auth/me")
async def get_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {"username": current_user["sub"], "message": "Authentication successful"}

@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    """Example of a protected route"""
    return {
        "message": f"Hello {current_user['sub']}, this is a protected route!",
        "user": current_user,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT) 