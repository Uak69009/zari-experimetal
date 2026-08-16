-- ============================================================================
-- ZARI.ai — RAG POSTGRESQL KNOWLEDGEBASE SCHEMA
-- ============================================================================

-- 1. DISEASES TABLE
CREATE TABLE IF NOT EXISTS diseases (
    disease_id VARCHAR(100) PRIMARY KEY,
    disease_class VARCHAR(100) NOT NULL,
    crop VARCHAR(100) NOT NULL,
    scientific_name VARCHAR(255) NOT NULL,
    pathogen_type VARCHAR(50) NOT NULL,
    common_names JSONB NOT NULL DEFAULT '{}'::jsonb,
    identity_status VARCHAR(50) NOT NULL DEFAULT 'verified',
    candidate_pathogens JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. PESTICIDES TABLE
CREATE TABLE IF NOT EXISTS pesticides (
    active_ingredient_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    frac_group VARCHAR(50),
    irac_group VARCHAR(50),
    type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. PRODUCTS TABLE
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(100) PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    manufacturer VARCHAR(255) NOT NULL,
    active_ingredients JSONB NOT NULL DEFAULT '[]'::jsonb,
    formulation VARCHAR(50),
    concentration VARCHAR(100),
    registered_in_pakistan BOOLEAN NOT NULL DEFAULT FALSE,
    registration_number VARCHAR(100),
    dpp_form VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. REGISTRATIONS TABLE
CREATE TABLE IF NOT EXISTS registrations (
    registration_id VARCHAR(100) PRIMARY KEY,
    product_id VARCHAR(100) REFERENCES products(product_id) ON DELETE CASCADE,
    crop VARCHAR(100) NOT NULL,
    target_disease VARCHAR(100) NOT NULL,
    label_rate VARCHAR(100),
    application_timing TEXT,
    phi_days INTEGER,
    rei_hours INTEGER,
    verification_status VARCHAR(50) NOT NULL DEFAULT 'unverified',
    source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. SOURCES TABLE
CREATE TABLE IF NOT EXISTS sources (
    source_id VARCHAR(100) PRIMARY KEY,
    organization VARCHAR(255) NOT NULL,
    document_title VARCHAR(255) NOT NULL,
    url TEXT,
    publication_date DATE,
    accessed_date DATE NOT NULL DEFAULT CURRENT_DATE,
    source_type VARCHAR(100) NOT NULL,
    evidence_level VARCHAR(20) NOT NULL,
    country VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- INDEXES FOR FAST RAG LOOKUPS
CREATE INDEX IF NOT EXISTS idx_diseases_crop ON diseases(crop);
CREATE INDEX IF NOT EXISTS idx_diseases_pathogen_type ON diseases(pathogen_type);
CREATE INDEX IF NOT EXISTS idx_pesticides_frac ON pesticides(frac_group);
CREATE INDEX IF NOT EXISTS idx_products_reg_pk ON products(registered_in_pakistan);
CREATE INDEX IF NOT EXISTS idx_registrations_crop_target ON registrations(crop, target_disease);
CREATE INDEX IF NOT EXISTS idx_sources_evidence ON sources(evidence_level);
