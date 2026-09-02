-- MTP Phase 1: Agentic Multimodal E-Commerce + Virtual Try-On
-- PostgreSQL 16+ recommended
-- Requires pgvector for embedding columns.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS commerce;
CREATE SCHEMA IF NOT EXISTS ai;
CREATE SCHEMA IF NOT EXISTS auth;

-- =========================
-- ENUMS
-- =========================

DO $$ BEGIN
    CREATE TYPE catalog.product_status AS ENUM ('DRAFT', 'ACTIVE', 'INACTIVE', 'ARCHIVED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE commerce.inventory_status AS ENUM ('AVAILABLE', 'RESERVED', 'PAYMENT_PENDING', 'SOLD', 'DAMAGED', 'RETURNED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE commerce.inventory_txn_type AS ENUM ('STOCK_IN', 'RESERVED', 'RELEASED', 'SOLD', 'RETURNED', 'DAMAGED', 'ADJUSTMENT');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ai.message_sender AS ENUM ('CUSTOMER', 'AGENT', 'HUMAN_AGENT', 'SYSTEM');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ai.message_type AS ENUM ('TEXT', 'IMAGE', 'VIDEO', 'DOCUMENT', 'PRODUCT', 'VTON_RESULT', 'PAYMENT', 'ORDER_UPDATE');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE commerce.order_status AS ENUM ('PENDING', 'PAYMENT_PENDING', 'PAID', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'CANCELLED', 'RETURNED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE commerce.payment_status AS ENUM ('CREATED', 'PENDING', 'SUCCESS', 'FAILED', 'REFUNDED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE commerce.delivery_status AS ENUM ('PENDING', 'READY_FOR_PICKUP', 'SHIPPED', 'IN_TRANSIT', 'DELIVERED', 'FAILED', 'RETURNED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ai.vton_status AS ENUM ('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE auth.staff_role AS ENUM ('ADMIN', 'STORE_MANAGER', 'STAFF');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- =========================
-- CATALOG
-- =========================

CREATE TABLE IF NOT EXISTS catalog.products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL DEFAULT 'SAMBALPURI_SAREE',
    fabric TEXT,
    primary_color TEXT,
    secondary_color TEXT,
    bandha_type TEXT,
    motif TEXT,
    thread_count INTEGER CHECK (thread_count IS NULL OR thread_count > 0),
    border_type TEXT,
    saree_length_cm NUMERIC(7,2) CHECK (saree_length_cm IS NULL OR saree_length_cm > 0),
    blouse_included BOOLEAN NOT NULL DEFAULT FALSE,
    weaving_region TEXT,
    occasion TEXT,
    weight_grams NUMERIC(8,2) CHECK (weight_grams IS NULL OR weight_grams > 0),
    status catalog.product_status NOT NULL DEFAULT 'DRAFT',
    public_slug TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.product_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    image_type TEXT NOT NULL,
    storage_provider TEXT NOT NULL DEFAULT 'r2',
    alt_text TEXT,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    is_private BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS commerce.product_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE RESTRICT,
    price NUMERIC(12,2) NOT NULL CHECK (price >= 0),
    currency CHAR(3) NOT NULL DEFAULT 'INR',
    effective_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    effective_to TIMESTAMPTZ,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (effective_to IS NULL OR effective_to > effective_from)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_product_current_price
    ON commerce.product_prices(product_id) WHERE is_current;

-- =========================
-- INVENTORY
-- =========================

CREATE TABLE IF NOT EXISTS commerce.inventory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE RESTRICT,
    inventory_code TEXT NOT NULL UNIQUE,
    status commerce.inventory_status NOT NULL DEFAULT 'AVAILABLE',
    location TEXT,
    purchase_date DATE,
    cost_price NUMERIC(12,2) CHECK (cost_price IS NULL OR cost_price >= 0),
    selling_price_override NUMERIC(12,2) CHECK (selling_price_override IS NULL OR selling_price_override >= 0),
    reserved_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS commerce.inventory_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_item_id UUID NOT NULL REFERENCES commerce.inventory_items(id) ON DELETE RESTRICT,
    transaction_type commerce.inventory_txn_type NOT NULL,
    reference_id UUID,
    performed_by UUID,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================
-- CUSTOMERS / AI MEMORY
-- =========================

CREATE TABLE IF NOT EXISTS ai.customers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    whatsapp_number TEXT NOT NULL UNIQUE,
    email TEXT,
    city TEXT,
    state TEXT,
    preferred_language TEXT DEFAULT 'en',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ai.customer_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES ai.customers(id) ON DELETE CASCADE,
    preference_type TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    source TEXT NOT NULL DEFAULT 'INFERRED',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES ai.customers(id) ON DELETE CASCADE,
    channel TEXT NOT NULL DEFAULT 'WHATSAPP',
    status TEXT NOT NULL DEFAULT 'OPEN',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES ai.conversations(id) ON DELETE CASCADE,
    sender_type ai.message_sender NOT NULL,
    message_type ai.message_type NOT NULL DEFAULT 'TEXT',
    content TEXT,
    media_url TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    whatsapp_message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai.recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES ai.customers(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES ai.conversations(id) ON DELETE SET NULL,
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE RESTRICT,
    inventory_item_id UUID REFERENCES commerce.inventory_items(id) ON DELETE SET NULL,
    reason TEXT,
    score NUMERIC(8,6),
    rank INTEGER CHECK (rank IS NULL OR rank > 0),
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 768 is an example dimension. Change it to the selected embedding model dimension before production.
CREATE TABLE IF NOT EXISTS catalog.product_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE CASCADE,
    embedding_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================
-- VTON
-- =========================

CREATE TABLE IF NOT EXISTS ai.vton_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES ai.customers(id) ON DELETE SET NULL,
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE RESTRICT,
    inventory_item_id UUID REFERENCES commerce.inventory_items(id) ON DELETE SET NULL,
    conversation_id UUID REFERENCES ai.conversations(id) ON DELETE SET NULL,
    customer_image_url TEXT NOT NULL,
    garment_image_url TEXT NOT NULL,
    status ai.vton_status NOT NULL DEFAULT 'QUEUED',
    model_name TEXT NOT NULL,
    model_version TEXT,
    prompt TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ai.generated_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vton_job_id UUID NOT NULL REFERENCES ai.vton_jobs(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    generation_type TEXT NOT NULL DEFAULT 'VTON_RESULT',
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================
-- COMMERCE
-- =========================

CREATE TABLE IF NOT EXISTS commerce.carts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES ai.customers(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS commerce.cart_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cart_id UUID NOT NULL REFERENCES commerce.carts(id) ON DELETE CASCADE,
    inventory_item_id UUID NOT NULL REFERENCES commerce.inventory_items(id) ON DELETE RESTRICT,
    price_at_addition NUMERIC(12,2) NOT NULL CHECK (price_at_addition >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (cart_id, inventory_item_id)
);

CREATE TABLE IF NOT EXISTS commerce.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES ai.customers(id) ON DELETE RESTRICT,
    order_number TEXT NOT NULL UNIQUE,
    status commerce.order_status NOT NULL DEFAULT 'PENDING',
    subtotal NUMERIC(12,2) NOT NULL CHECK (subtotal >= 0),
    discount NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (discount >= 0),
    shipping_fee NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (shipping_fee >= 0),
    total_amount NUMERIC(12,2) NOT NULL CHECK (total_amount >= 0),
    currency CHAR(3) NOT NULL DEFAULT 'INR',
    shipping_address JSONB NOT NULL,
    billing_address JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS commerce.order_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES commerce.orders(id) ON DELETE CASCADE,
    inventory_item_id UUID NOT NULL REFERENCES commerce.inventory_items(id) ON DELETE RESTRICT,
    product_id UUID NOT NULL REFERENCES catalog.products(id) ON DELETE RESTRICT,
    quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity = 1),
    unit_price NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    total_price NUMERIC(12,2) NOT NULL CHECK (total_price >= 0)
);

CREATE TABLE IF NOT EXISTS commerce.payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES commerce.orders(id) ON DELETE RESTRICT,
    provider TEXT NOT NULL,
    transaction_id TEXT UNIQUE,
    amount NUMERIC(12,2) NOT NULL CHECK (amount >= 0),
    currency CHAR(3) NOT NULL DEFAULT 'INR',
    status commerce.payment_status NOT NULL DEFAULT 'CREATED',
    payment_method TEXT,
    paid_at TIMESTAMPTZ,
    provider_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS commerce.deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL UNIQUE REFERENCES commerce.orders(id) ON DELETE RESTRICT,
    courier_provider TEXT,
    tracking_number TEXT,
    status commerce.delivery_status NOT NULL DEFAULT 'PENDING',
    pickup_address JSONB,
    delivery_address JSONB NOT NULL,
    estimated_delivery_date DATE,
    shipped_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================
-- STAFF / AUDIT
-- =========================

CREATE TABLE IF NOT EXISTS auth.staff_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    phone TEXT UNIQUE,
    email TEXT UNIQUE,
    role auth.staff_role NOT NULL DEFAULT 'STAFF',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS auth.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.staff_users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id UUID,
    old_value JSONB,
    new_value JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================
-- INDEXES
-- =========================

CREATE INDEX IF NOT EXISTS ix_products_status ON catalog.products(status);
CREATE INDEX IF NOT EXISTS ix_products_bandha ON catalog.products(bandha_type);
CREATE INDEX IF NOT EXISTS ix_products_thread_count ON catalog.products(thread_count);
CREATE INDEX IF NOT EXISTS ix_products_primary_color ON catalog.products(primary_color);
CREATE INDEX IF NOT EXISTS ix_inventory_product_status ON commerce.inventory_items(product_id, status);
CREATE INDEX IF NOT EXISTS ix_inventory_reserved_until ON commerce.inventory_items(reserved_until) WHERE status = 'RESERVED';
CREATE INDEX IF NOT EXISTS ix_messages_conversation_created ON ai.messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS ix_recommendations_customer_created ON ai.recommendations(customer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_vton_customer_created ON ai.vton_jobs(customer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_orders_customer_created ON commerce.orders(customer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_orders_status ON commerce.orders(status);
CREATE INDEX IF NOT EXISTS ix_payments_order ON commerce.payments(order_id);
CREATE INDEX IF NOT EXISTS ix_audit_entity ON auth.audit_logs(entity_type, entity_id, created_at DESC);

-- Approximate nearest-neighbor index for pgvector. Tune after measuring data scale.
CREATE INDEX IF NOT EXISTS ix_product_embeddings_hnsw
    ON catalog.product_embeddings USING hnsw (embedding vector_cosine_ops);

-- =========================
-- PUBLIC CATALOG VIEW
-- =========================
-- Intentionally excludes internal price and inventory data.
CREATE OR REPLACE VIEW catalog.public_products AS
SELECT
    p.id,
    p.sku,
    p.name,
    p.description,
    p.category,
    p.fabric,
    p.primary_color,
    p.secondary_color,
    p.bandha_type,
    p.motif,
    p.thread_count,
    p.border_type,
    p.saree_length_cm,
    p.blouse_included,
    p.weaving_region,
    p.occasion,
    p.weight_grams,
    p.public_slug,
    p.status,
    p.created_at,
    p.updated_at
FROM catalog.products p
WHERE p.status = 'ACTIVE';

-- =========================
-- SECURITY NOTES
-- =========================
-- 1. Keep product_prices, inventory, payments and audit tables behind authenticated backend roles.
-- 2. Never expose catalog.public_products through an endpoint that joins price/inventory data.
-- 3. Store customer/VTON images in private object storage and issue short-lived signed URLs.
-- 4. Never store card numbers/CVV; only store payment-provider transaction identifiers.
-- 5. Add PostgreSQL RLS policies once application roles are defined.
-- 6. Perform inventory reservation + order creation inside a DB transaction using row-level locking.
