# Database — Beginner Guide

This document explains the Phase 1 database in simple terms.

## What is PostgreSQL?

PostgreSQL is the main database where the application stores structured business information such as products, customers, inventory, orders, and payments.

Think of it as a collection of connected tables rather than one giant spreadsheet.

## Why separate tables?

Each table stores one type of information. Relationships between tables are represented with IDs.

For example:

```text
products
   |
   | product_id
   v
inventory_items
```

A product describes *what a saree is*, while an inventory item describes *which physical piece we currently have*.

## Key concepts

### Primary Key (PK)
A unique identifier for a row.

### Foreign Key (FK)
A reference to a row in another table.

### Index
A structure that makes common searches faster.

### Transaction
A group of database operations that succeed together or fail together.

### ENUM
A restricted set of valid states, useful for things like `AVAILABLE`, `SOLD`, and `CANCELLED`.

### JSONB
PostgreSQL's flexible JSON data type. We use it for some data whose structure may evolve, such as addresses and provider payloads.

### pgvector
A PostgreSQL extension that lets us store and search numerical embeddings for multimodal similarity retrieval.

## Price privacy

The public website reads from `catalog.public_products`, which deliberately excludes private price and inventory fields.

The WhatsApp agent uses authenticated backend tools to access private pricing and inventory data.

## Important rule

The LLM should never receive unrestricted access to the database. The application will expose controlled tools that validate inputs and enforce authorization.
