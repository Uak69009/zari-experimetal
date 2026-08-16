"""ZARI.ai — PostgreSQL Schema Setup for RAG Knowledgebase.

This script creates and manages the PostgreSQL database schema for ZARI RAG:
1. diseases        : Disease taxonomy, crop, common names (JSONB), candidate pathogens
2. pesticides      : Active ingredients, FRAC/IRAC codes, type
3. products        : Commercial products, formulations, Pakistan registration status
4. registrations   : Label application rates, PHI days, REI, crops, diseases
5. sources        : Evidence sources, document titles, evidence levels (A1-D)

Environment Variables Used:
- DATABASE_URL or (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    psycopg2 = None

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAG_DIR = SCRIPT_DIR
SCHEMA_SQL_PATH = RAG_DIR / "schema.sql"

# Master SQL DDL Schema Statements
DDL_SCHEMA = """-- ============================================================================
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
"""


def get_db_credentials() -> dict[str, str]:
    """Extract database credentials from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return {"url": database_url}

    return {
        "dbname": os.getenv("POSTGRES_DB", "zari_rag"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
    }


def main() -> None:
    print("=" * 70)
    print("  ZARI.ai — RAG POSTGRESQL SCHEMA SETUP")
    print("=" * 70)

    if psycopg2 is None:
        raise ImportError("psycopg2-binary module required. Run: pip install psycopg2-binary")

    RAG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save SQL file
    SCHEMA_SQL_PATH.write_text(DDL_SCHEMA, encoding="utf-8")
    print(f"\n✓ Saved standalone DDL schema to: {SCHEMA_SQL_PATH}")

    creds = get_db_credentials()
    conn = None
    target_db = creds.get("dbname", "zari_rag")

    print("\nAttempting PostgreSQL connection...")
    try:
        if "url" in creds:
            print(f"Connecting via DATABASE_URL...")
            conn = psycopg2.connect(creds["url"])
        else:
            host = creds["host"]
            port = creds["port"]
            user = creds["user"]
            print(f"Target DB: {user}@{host}:{port}/{target_db}")

            # First connect to default 'postgres' database to check/create target database
            try:
                sys_conn = psycopg2.connect(
                    dbname="postgres", user=user, password=creds["password"], host=host, port=port
                )
                sys_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = sys_conn.cursor()
                cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s;", (target_db,))
                exists = cursor.fetchone()
                if not exists:
                    print(f"Creating database '{target_db}'...")
                    cursor.execute(f'CREATE DATABASE "{target_db}";')
                    print(f"✓ Database '{target_db}' created successfully.")
                cursor.close()
                sys_conn.close()
            except Exception as e:
                print(f"Note (default db connect): {e}")

            # Connect to target database
            conn = psycopg2.connect(
                dbname=target_db, user=user, password=creds["password"], host=host, port=port
            )

        # Execute DDL
        cur = conn.cursor()
        cur.execute(DDL_SCHEMA)
        conn.commit()
        cur.close()
        conn.close()

        print("\n" + "=" * 70)
        print("  SCHEMA CREATION CONFIRMATION")
        print("=" * 70)
        print("✓ Connected to PostgreSQL successfully!")
        print("✓ Created/Verified the following 5 tables in database:")
        print("   1. diseases      (disease_id, disease_class, crop, scientific_name, pathogen_type, common_names, identity_status, candidate_pathogens)")
        print("   2. pesticides    (active_ingredient_id, name, frac_group, irac_group, type)")
        print("   3. products      (product_id, product_name, manufacturer, active_ingredients, formulation, concentration, registered_in_pakistan, registration_number, dpp_form)")
        print("   4. registrations (registration_id, product_id, crop, target_disease, label_rate, application_timing, phi_days, rei_hours, verification_status, source_url)")
        print("   5. sources      (source_id, organization, document_title, url, publication_date, accessed_date, source_type, evidence_level, country)")
        print("\n✅ POSTGRESQL SCHEMA SETUP COMPLETE!")

    except Exception as e:
        print("\n" + "!" * 70)
        print("  POSTGRESQL CONNECTION NOTICE")
        print("!" * 70)
        print(f"Could not connect to PostgreSQL instance: {e}")
        print(f"✓ DDL Schema SQL file successfully generated at:")
        print(f"   {SCHEMA_SQL_PATH}")
        print("\nTo initialize when PostgreSQL is running, execute:")
        print(f"   psql -h <host> -U <user> -d <dbname> -f \"{SCHEMA_SQL_PATH}\"")
        print("!" * 70)


if __name__ == "__main__":
    main()
