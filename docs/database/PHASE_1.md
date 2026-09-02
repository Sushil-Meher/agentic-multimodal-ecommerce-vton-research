# Phase 1 — Database Implementation Plan

## Goal

Create a secure and maintainable PostgreSQL foundation for the Sambalpuri AI commerce system.

## Step sequence

1. Install Docker Desktop.
2. Start PostgreSQL + pgvector with Docker Compose.
3. Connect using a database client such as pgAdmin or DBeaver.
4. Run `docs/database/schema.sql`.
5. Confirm schemas and tables were created.
6. Insert a small set of fake test sarees.
7. Query the public catalog view and verify that price is not returned.
8. Test an inventory reservation transaction.
9. Commit the verified Phase 1 implementation.

## Learning objective

By the end of this phase, understand:

- databases vs tables
- primary and foreign keys
- SQL `SELECT`, `INSERT`, `UPDATE`, and `DELETE`
- joins
- indexes
- transactions
- constraints
- PostgreSQL schemas
- vector embeddings at a high level
