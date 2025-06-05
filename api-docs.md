# Strategy AI Backend API Documentation

## Base URL
```
https://your-railway-app.railway.app
```

## Authentication
Most endpoints require an admin key header:
```
X-Admin-Key: your-admin-key
```

## API Endpoints

### Chat Endpoints

#### POST /api/chat
Basic chat interaction endpoint.

Request:
```typescript
{
  message: string;
  sector: string;
  use_case?: string;
  session_id: string;
  user_type?: string;
}
```

Response:
```typescript
{
  response: string;
  sources: Array<{
    document_title: string;
    source: string;
    relevance_score: number;
    chunk_preview: string;
  }>;
  confidence: number;
  suggested_use_case: string;
  timestamp: string;
  chat_log_id: string;
}
```

#### POST /api/chat/advanced
Enhanced chat with specialized AI agents.

Request: Same as /api/chat

Response:
```typescript
{
  response: string;
  sources: Array<{
    document_title: string;
    source: string;
    relevance_score: number;
    chunk_preview: string;
  }>;
  confidence: number;
  suggested_use_case: string;
  timestamp: string;
  chat_log_id: string;
  enhanced_features: {
    agents_used: string[];
    response_type: string;
    analysis_available: boolean;
  };
}
```

### Document Management

#### POST /api/documents/upload
Upload and process a new document.

Request:
- FormData with:
  - file: File
  - sector: string
  - use_case?: string
  - admin_key: string

Response:
```typescript
{
  success: boolean;
  document_id: string;
  message: string;
}
```

#### GET /api/documents
List available documents.

Query Parameters:
- sector?: string
- use_case?: string
- source_type?: string
- search?: string
- min_rating?: number
- limit?: number (default: 50)
- offset?: number (default: 0)

Response:
```typescript
{
  documents: Array<{
    id: string;
    filename: string;
    sector: string;
    use_case: string;
    metadata: {
      title: string;
      source: string;
      date: string;
      summary: string;
    };
    status: string;
    created_at: string;
  }>;
  total_count: number;
  limit: number;
  offset: number;
  has_more: boolean;
}
```

### Report Generation

#### POST /api/reports/generate
Generate a comprehensive report.

Request:
- FormData with:
  - report_type: string
  - sector: string (default: "General")
  - format: "pdf" | "docx" | "both"
  - title?: string
  - scope: "comprehensive" | "summary"
  - admin_key: string

Response:
```typescript
{
  success: boolean;
  report_id: string;
  download_urls: string[];
  metadata: {
    report_id: string;
    report_type: string;
    generated_at: string;
    parameters: Record<string, any>;
    files: Array<{
      format: string;
      filename: string;
    }>;
  };
  message: string;
}
```

#### POST /api/reports/generate-stream
Generate report with real-time updates via Server-Sent Events (SSE).

Request: Same as /api/reports/generate

Response Stream Events:
```typescript
// Initial Status
{
  status: "starting";
  message: string;
}

// Section Updates
{
  status: "generating";
  section: string;
  progress: number;
}

// Section Completion
{
  status: "section_complete";
  section: string;
  content: string;
  progress: number;
}

// Final Completion
{
  status: "complete";
  report_id: string;
  download_urls: string[];
}

// Error Event
{
  error: string;
}
```

### WebSocket Endpoints

#### WS /ws/document/{document_id}
Real-time document processing updates.

Connection:
```typescript
const ws = new WebSocket(`wss://your-railway-app.railway.app/ws/document/${documentId}`);
```

Message Events:
```typescript
// Processing Update
{
  type: "processing_update";
  document_id: string;
  status: string;
  progress: number;
  message: string;
  timestamp: string;
}

// Error Event
{
  type: "error";
  document_id: string;
  error: string;
  status_code: number;
  timestamp: string;
}
```

### Progress Monitoring

#### GET /api/admin/active-processes
Get all active document processing and report generation tasks.

Response:
```typescript
{
  active_documents: Record<string, {
    status: string;
    progress: number;
    created_at: string;
    message?: string;
  }>;
  active_reports: Record<string, {
    status: string;
    progress: number;
    created_at: string;
    completed_sections: string[];
  }>;
  total_active_processes: number;
  timestamp: string;
}
```

## Status Codes
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Rate Limiting
- WebSocket updates: 100ms minimum interval between updates
- Maximum 5 connection attempts per document within 5 minutes 