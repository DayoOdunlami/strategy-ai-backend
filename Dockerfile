FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies one by one
RUN pip install fastapi==0.104.1
RUN pip install "uvicorn[standard]==0.24.0"
RUN pip install pydantic==2.5.0
RUN pip install python-multipart==0.0.6
RUN pip install supabase==2.3.0
RUN pip install pinecone-client==3.0.0
RUN pip install openai==1.3.7
RUN pip install tiktoken==0.5.2
RUN pip install python-dotenv==1.0.0
RUN pip install requests==2.31.0
RUN pip install httpx==0.25.2

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 