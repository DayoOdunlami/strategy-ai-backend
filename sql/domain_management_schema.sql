-- Domain Management System Schema for Supabase
-- Run this in your Supabase SQL editor

-- Enable RLS (Row Level Security) if needed
-- ALTER TABLE domains ENABLE ROW LEVEL SECURITY;

-- First, let's check if the existing use_cases table has incompatible schema
-- If so, we'll need to create a new table with our desired schema

-- Drop existing tables if they have incompatible schemas (backup first if needed)
DO $$
BEGIN
    -- Check if use_cases table exists with incompatible schema
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'use_cases' AND column_name = 'sector') THEN
        -- The existing table has a 'sector' column which conflicts with our design
        -- We'll rename it to preserve data, then create our new schema
        RAISE NOTICE 'Found existing use_cases table with incompatible schema';
        
        -- Only rename if backup doesn't already exist
        IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'use_cases_backup') THEN
            ALTER TABLE use_cases RENAME TO use_cases_backup;
            RAISE NOTICE 'Renamed existing use_cases to use_cases_backup';
        END IF;
    END IF;
END $$;

-- 1. Domains table
CREATE TABLE IF NOT EXISTS domains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    color VARCHAR(7), -- hex color code like #ff0000
    icon VARCHAR(50), -- icon name or unicode
    is_active BOOLEAN DEFAULT true,
    document_count INTEGER DEFAULT 0,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add unique constraint on name if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'domains_name_unique') THEN
        ALTER TABLE domains ADD CONSTRAINT domains_name_unique UNIQUE (name);
    END IF;
END $$;

-- Add missing columns to domains table if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'domains' AND column_name = 'color') THEN
        ALTER TABLE domains ADD COLUMN color VARCHAR(7);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'domains' AND column_name = 'icon') THEN
        ALTER TABLE domains ADD COLUMN icon VARCHAR(50);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'domains' AND column_name = 'is_active') THEN
        ALTER TABLE domains ADD COLUMN is_active BOOLEAN DEFAULT true;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'domains' AND column_name = 'document_count') THEN
        ALTER TABLE domains ADD COLUMN document_count INTEGER DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'domains' AND column_name = 'sort_order') THEN
        ALTER TABLE domains ADD COLUMN sort_order INTEGER DEFAULT 0;
    END IF;
END $$;

-- 2. Use Cases table (with domain_id foreign key for hierarchical relationship)
CREATE TABLE IF NOT EXISTS use_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50), -- Strategic, Operational, etc.
    domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT true,
    document_count INTEGER DEFAULT 0,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add unique constraint on name if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'use_cases_name_unique') THEN
        ALTER TABLE use_cases ADD CONSTRAINT use_cases_name_unique UNIQUE (name);
    END IF;
END $$;

-- Add missing columns to use_cases table if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'use_cases' AND column_name = 'category') THEN
        ALTER TABLE use_cases ADD COLUMN category VARCHAR(50);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'use_cases' AND column_name = 'domain_id') THEN
        ALTER TABLE use_cases ADD COLUMN domain_id UUID REFERENCES domains(id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'use_cases' AND column_name = 'is_active') THEN
        ALTER TABLE use_cases ADD COLUMN is_active BOOLEAN DEFAULT true;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'use_cases' AND column_name = 'document_count') THEN
        ALTER TABLE use_cases ADD COLUMN document_count INTEGER DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'use_cases' AND column_name = 'sort_order') THEN
        ALTER TABLE use_cases ADD COLUMN sort_order INTEGER DEFAULT 0;
    END IF;
END $$;

-- 3. Prompt Templates table (optional - for future use)
CREATE TABLE IF NOT EXISTS prompt_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    template_text TEXT NOT NULL,
    use_case_id UUID REFERENCES use_cases(id) ON DELETE CASCADE,
    domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    variables JSONB, -- Array of variable names like ["document_type", "sector"]
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_domains_name ON domains(name);
CREATE INDEX IF NOT EXISTS idx_domains_active ON domains(is_active);
CREATE INDEX IF NOT EXISTS idx_use_cases_name ON use_cases(name);
CREATE INDEX IF NOT EXISTS idx_use_cases_category ON use_cases(category);
CREATE INDEX IF NOT EXISTS idx_use_cases_domain_id ON use_cases(domain_id);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_use_case ON prompt_templates(use_case_id);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_domain ON prompt_templates(domain_id);

-- 5. Update triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_domains_updated_at BEFORE UPDATE ON domains 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_use_cases_updated_at BEFORE UPDATE ON use_cases 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prompt_templates_updated_at BEFORE UPDATE ON prompt_templates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 6. Insert existing hardcoded data
INSERT INTO domains (name, description, color, icon) VALUES
('Rail & Transit', 'Railway infrastructure, stations, and transit systems', '#8B5CF6', '🚆'),
('Highway & Roads', 'Road infrastructure, traffic management, and highway systems', '#F59E0B', '🛣️'),
('Maritime', 'Ports, shipping, and maritime transportation systems', '#06B6D4', '⚓'),
('AI & Technology', 'Machine Learning, AI systems, and emerging technologies', '#3B82F6', '🤖'),
('General', 'Cross-cutting topics and general strategy documents', '#6B7280', '📋')
ON CONFLICT (name) DO NOTHING;

-- Get domain IDs for use case insertion
DO $$
DECLARE
    rail_domain_id UUID;
    highway_domain_id UUID;
    maritime_domain_id UUID;
    ai_domain_id UUID;
    general_domain_id UUID;
BEGIN
    -- Get domain IDs
    SELECT id INTO rail_domain_id FROM domains WHERE name = 'Rail & Transit';
    SELECT id INTO highway_domain_id FROM domains WHERE name = 'Highway & Roads';
    SELECT id INTO maritime_domain_id FROM domains WHERE name = 'Maritime';
    SELECT id INTO ai_domain_id FROM domains WHERE name = 'AI & Technology';
    SELECT id INTO general_domain_id FROM domains WHERE name = 'General';

    -- Insert use cases for Rail & Transit
    INSERT INTO use_cases (name, description, category, domain_id) VALUES
    ('Strategy Development', 'Strategic planning and high-level decision making', 'Strategic', rail_domain_id),
    ('Infrastructure Planning', 'Plan and assess infrastructure projects', 'Operational', rail_domain_id),
    ('Safety & Compliance', 'Safety protocols and regulatory compliance', 'Assessment', rail_domain_id)
    ON CONFLICT (name) DO NOTHING;

    -- Insert use cases for Highway & Roads
    INSERT INTO use_cases (name, description, category, domain_id) VALUES
    ('Traffic Management', 'Optimize traffic flow and management systems', 'Operational', highway_domain_id),
    ('Smart Infrastructure', 'IoT and smart road technologies', 'Strategic', highway_domain_id)
    ON CONFLICT (name) DO NOTHING;

    -- Insert use cases for Maritime
    INSERT INTO use_cases (name, description, category, domain_id) VALUES
    ('Port Operations', 'Harbor and port facility management', 'Operational', maritime_domain_id),
    ('Shipping Analytics', 'Maritime logistics and shipping optimization', 'Analysis', maritime_domain_id)
    ON CONFLICT (name) DO NOTHING;

    -- Insert general use cases that can be used across domains
    INSERT INTO use_cases (name, description, category, domain_id) VALUES
    ('General Analysis', 'General purpose analysis and research', 'General', general_domain_id),
    ('Project Review', 'Project reviews and assessments', 'Assessment', general_domain_id)
    ON CONFLICT (name) DO NOTHING;
END $$;

-- 7. Verification queries (optional - for testing)
-- Run these to verify the schema was created correctly:
-- SELECT 'Domains' as table_name, COUNT(*) as record_count FROM domains;
-- SELECT 'Use Cases' as table_name, COUNT(*) as record_count FROM use_cases;
-- SELECT d.name as domain, COUNT(uc.id) as use_case_count FROM domains d LEFT JOIN use_cases uc ON d.id = uc.domain_id GROUP BY d.id, d.name ORDER BY d.name; 