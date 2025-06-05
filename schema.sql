-- Supabase Database Schema for Strategy AI Backend
-- Run this SQL in your Supabase SQL editor to create all tables

-- ============================================================================
-- SECTORS TABLE (Primary Dropdown)
-- ============================================================================
CREATE TABLE sectors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default sectors
INSERT INTO sectors (name, description) VALUES
    ('Rail', 'Rail sector strategic documents and use cases'),
    ('Maritime', 'Maritime sector strategic documents and use cases'),
    ('Highways', 'Highway and roads sector strategic documents and use cases'),
    ('General', 'General cross-sector strategic documents');

-- ============================================================================
-- USE CASES TABLE (Secondary Dropdown with Custom Prompts)
-- ============================================================================
CREATE TABLE use_cases (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    sector VARCHAR(100) NOT NULL REFERENCES sectors(name) ON UPDATE CASCADE,
    tags TEXT,
    prompt_template TEXT, -- Custom prompt templates for each use case
    description TEXT,
    example_queries JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name, sector)
);

-- Insert default use cases for Rail sector
INSERT INTO use_cases (name, sector, tags, description) VALUES
    ('Quick Playbook Answers', 'Rail', 'playbook,guidance,quick', 'Direct questions about processes, guidelines, and standards'),
    ('Lessons Learned', 'Rail', 'lessons,experience,insights', 'Learning from past projects and experiences'),
    ('Project Review / MOT', 'Rail', 'review,assessment,health', 'Health checks, status reviews, and project assessments'),
    ('TRL / RIRL Mapping', 'Rail', 'readiness,technology,assessment', 'Technology readiness level assessments'),
    ('Project Similarity', 'Rail', 'similarity,matching,comparison', 'Finding similar past projects and case studies'),
    ('Change Management', 'Rail', 'transition,handover,change', 'Transitions, handovers, and organizational changes'),
    ('Product Acceptance', 'Rail', 'acceptance,approval,governance', 'Approval processes, compliance, and governance');

-- ============================================================================
-- DOCUMENTS TABLE (Main Document Storage)
-- ============================================================================
CREATE TABLE documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    filename VARCHAR(255), -- Original filename for uploaded files
    sector VARCHAR(100) NOT NULL REFERENCES sectors(name) ON UPDATE CASCADE,
    use_case VARCHAR(200) REFERENCES use_cases(name) ON UPDATE CASCADE,
    tags TEXT,
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('file', 'url', 'csv_import')),
    source_url TEXT, -- URL for scraped content
    status VARCHAR(50) DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed')),
    chunk_count INTEGER DEFAULT 0,
    file_size_bytes BIGINT,
    error_message TEXT, -- Store processing errors
    metadata JSONB DEFAULT '{}'::jsonb, -- Additional custom metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for better query performance
CREATE INDEX idx_documents_sector ON documents(sector);
CREATE INDEX idx_documents_use_case ON documents(use_case);
CREATE INDEX idx_documents_source_type ON documents(source_type);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_title_search ON documents USING gin(to_tsvector('english', title));

-- ============================================================================
-- CHUNKS TABLE (Document Chunks for Vector Storage)
-- ============================================================================
CREATE TABLE chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb, -- AI-generated metadata (topic, summary, etc.)
    pinecone_id VARCHAR(255) UNIQUE, -- Reference to Pinecone vector
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- Add indexes for chunk queries
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_pinecone_id ON chunks(pinecone_id);
CREATE INDEX idx_chunks_text_search ON chunks USING gin(to_tsvector('english', text));

-- ============================================================================
-- CHAT LOGS TABLE (User Interaction Analytics)
-- ============================================================================
CREATE TABLE chat_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id VARCHAR(255), -- Track user sessions
    query TEXT NOT NULL,
    response TEXT,
    sources_used JSONB DEFAULT '[]'::jsonb, -- Document IDs used in response
    sector VARCHAR(100),
    use_case VARCHAR(200),
    user_type VARCHAR(50) DEFAULT 'public' CHECK (user_type IN ('public', 'admin', 'analyst')),
    confidence_score FLOAT,
    response_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for analytics queries
CREATE INDEX idx_chat_logs_session_id ON chat_logs(session_id);
CREATE INDEX idx_chat_logs_sector ON chat_logs(sector);
CREATE INDEX idx_chat_logs_use_case ON chat_logs(use_case);
CREATE INDEX idx_chat_logs_timestamp ON chat_logs(timestamp);

-- ============================================================================
-- USER FEEDBACK TABLE (NEW - User Rating & Feedback System)
-- ============================================================================
CREATE TABLE user_feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chat_log_id UUID REFERENCES chat_logs(id),
    document_id UUID REFERENCES documents(id),
    session_id VARCHAR(255),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5), -- 1-5 stars
    feedback_type VARCHAR(50) DEFAULT 'general' CHECK (feedback_type IN ('response_quality', 'source_relevance', 'general')),
    comment TEXT,
    helpful BOOLEAN, -- Quick thumbs up/down
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for feedback analytics
CREATE INDEX idx_user_feedback_rating ON user_feedback(rating);
CREATE INDEX idx_user_feedback_type ON user_feedback(feedback_type);
CREATE INDEX idx_user_feedback_helpful ON user_feedback(helpful);
CREATE INDEX idx_user_feedback_created_at ON user_feedback(created_at);
CREATE INDEX idx_user_feedback_chat_log ON user_feedback(chat_log_id);
CREATE INDEX idx_user_feedback_document ON user_feedback(document_id);

-- ============================================================================
-- BACKGROUND JOBS TABLES (Task Status Tracking)
-- ============================================================================

-- Scraping Jobs Status
CREATE TABLE scraping_jobs (
    job_id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    progress FLOAT DEFAULT 0.0,
    pages_scraped INTEGER DEFAULT 0,
    max_pages INTEGER DEFAULT 10,
    current_page TEXT,
    sector VARCHAR(100),
    use_case VARCHAR(200),
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- CSV Import Jobs Status
CREATE TABLE import_jobs (
    import_id UUID PRIMARY KEY,
    filename VARCHAR(255),
    status VARCHAR(50) DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    stats JSONB DEFAULT '{}'::jsonb, -- {"updated": 0, "created": 0, "failed": 0}
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- USER SESSIONS TABLE (Optional - for advanced analytics)
-- ============================================================================
CREATE TABLE user_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_type VARCHAR(50) DEFAULT 'public',
    first_query_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    query_count INTEGER DEFAULT 0,
    sectors_used JSONB DEFAULT '[]'::jsonb,
    use_cases_used JSONB DEFAULT '[]'::jsonb,
    user_agent TEXT,
    ip_address INET
);

-- ============================================================================
-- SYSTEM SETTINGS TABLE (Configuration Management)
-- ============================================================================
CREATE TABLE system_settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    category VARCHAR(50) DEFAULT 'general',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(100)
);

-- Insert default system settings
INSERT INTO system_settings (key, value, description, category) VALUES
    ('max_file_size_mb', '50', 'Maximum file size for uploads in MB', 'uploads'),
    ('allowed_file_types', '[".pdf", ".docx", ".txt", ".csv", ".md"]', 'Allowed file extensions for upload', 'uploads'),
    ('default_chunk_size', '1000', 'Default chunk size for document processing', 'processing'),
    ('default_chunk_overlap', '200', 'Default chunk overlap for document processing', 'processing'),
    ('vector_search_top_k', '8', 'Default number of results for vector search', 'search'),
    ('analytics_retention_days', '365', 'Days to retain analytics data', 'analytics'),
    ('auto_tag_documents', 'true', 'Automatically generate tags for uploaded documents', 'processing');

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

-- Add triggers for updated_at fields
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_use_cases_updated_at BEFORE UPDATE ON use_cases 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_scraping_jobs_updated_at BEFORE UPDATE ON scraping_jobs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_import_jobs_updated_at BEFORE UPDATE ON import_jobs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically update document chunk count
CREATE OR REPLACE FUNCTION update_document_chunk_count()
RETURNS TRIGGER AS $
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documents 
        SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE document_id = NEW.document_id)
        WHERE id = NEW.document_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documents 
        SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE document_id = OLD.document_id)
        WHERE id = OLD.document_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$ language 'plpgsql';

-- Trigger to keep chunk count updated
CREATE TRIGGER update_chunk_count_on_insert AFTER INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_chunk_count();

CREATE TRIGGER update_chunk_count_on_delete AFTER DELETE ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_chunk_count();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES (Including Feedback)
-- ============================================================================

-- View for document summary with chunk info and feedback
CREATE VIEW document_summary AS
SELECT 
    d.id,
    d.title,
    d.filename,
    d.sector,
    d.use_case,
    d.tags,
    d.source_type,
    d.source_url,
    d.status,
    d.chunk_count,
    d.created_at,
    d.updated_at,
    COUNT(DISTINCT cl.id) as query_count,
    MAX(cl.timestamp) as last_queried,
    COUNT(DISTINCT uf.id) as feedback_count,
    AVG(uf.rating) as average_rating
FROM documents d
LEFT JOIN chat_logs cl ON d.id = ANY(
    SELECT jsonb_array_elements_text(cl.sources_used)::uuid
)
LEFT JOIN user_feedback uf ON d.id = uf.document_id
GROUP BY d.id, d.title, d.filename, d.sector, d.use_case, d.tags, d.source_type, d.source_url, d.status, d.chunk_count, d.created_at, d.updated_at;

-- View for feedback analytics
CREATE VIEW feedback_analytics AS
SELECT 
    DATE(created_at) as feedback_date,
    feedback_type,
    COUNT(*) as total_feedback,
    AVG(rating) as average_rating,
    COUNT(CASE WHEN helpful = true THEN 1 END) as helpful_count,
    COUNT(CASE WHEN helpful = false THEN 1 END) as not_helpful_count,
    COUNT(CASE WHEN comment IS NOT NULL AND LENGTH(comment) > 0 THEN 1 END) as with_comments
FROM user_feedback 
GROUP BY DATE(created_at), feedback_type
ORDER BY feedback_date DESC;

-- View for sector analytics with feedback
CREATE VIEW sector_analytics AS
SELECT 
    s.name as sector_name,
    s.description,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(DISTINCT uc.id) as use_case_count,
    COUNT(DISTINCT cl.id) as query_count,
    COUNT(DISTINCT uf.id) as feedback_count,
    AVG(uf.rating) as average_rating,
    MAX(cl.timestamp) as last_activity
FROM sectors s
LEFT JOIN documents d ON s.name = d.sector
LEFT JOIN use_cases uc ON s.name = uc.sector
LEFT JOIN chat_logs cl ON s.name = cl.sector
LEFT JOIN user_feedback uf ON cl.id = uf.chat_log_id
GROUP BY s.name, s.description;

-- ============================================================================
-- SAMPLE DATA FOR TESTING (Optional)
-- ============================================================================

-- Insert sample document for testing
INSERT INTO documents (id, title, filename, sector, use_case, tags, source_type, status, chunk_count) VALUES
    (gen_random_uuid(), 'Sample Rail Strategy Document', 'rail_strategy_2024.pdf', 'Rail', 'Quick Playbook Answers', 'strategy,governance,rail', 'file', 'completed', 5);

-- Insert sample chat log
INSERT INTO chat_logs (session_id, query, response, sector, use_case, user_type) VALUES
    ('test_session_1', 'What is the rail modernization strategy?', 'Based on the strategy documents...', 'Rail', 'Quick Playbook Answers', 'public');

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE sectors IS 'Primary classification system for documents (Rail, Maritime, etc.)';
COMMENT ON TABLE use_cases IS 'Secondary classification with custom prompt templates for AI responses';
COMMENT ON TABLE documents IS 'Main document storage with metadata and processing status';
COMMENT ON TABLE chunks IS 'Document chunks for vector storage and retrieval';
COMMENT ON TABLE chat_logs IS 'User interaction logs for analytics and improvement';
COMMENT ON TABLE user_feedback IS 'User ratings and feedback on AI responses and document quality';
COMMENT ON TABLE scraping_jobs IS 'Status tracking for web scraping background tasks';
COMMENT ON TABLE import_jobs IS 'Status tracking for CSV import background tasks';

COMMENT ON COLUMN use_cases.prompt_template IS 'Custom prompt template for AI responses in this use case';
COMMENT ON COLUMN documents.metadata IS 'JSON field for additional custom metadata';
COMMENT ON COLUMN chunks.pinecone_id IS 'Reference ID for the corresponding vector in Pinecone';
COMMENT ON COLUMN chat_logs.sources_used IS 'JSON array of document IDs used to generate the response';
COMMENT ON COLUMN user_feedback.rating IS '1-5 star rating for AI response or document quality';
COMMENT ON COLUMN user_feedback.helpful IS 'Quick thumbs up/down feedback boolean';