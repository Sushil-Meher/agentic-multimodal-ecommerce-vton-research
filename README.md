# Agentic Multimodal E-Commerce & Virtual Try-On Research

## Research Title

**An Agentic Multimodal Framework for Personalized E-Commerce and Virtual Try-On of Traditional Indian Handloom Fashion**

### Real-world deployment

Sambalpuri handloom saree commerce using WhatsApp as the primary conversational sales channel and a public visual catalog website.

---

## Project Vision

This project develops and evaluates an agentic, multimodal AI commerce platform for traditional Indian handloom fashion.

The system is designed around a real Sambalpuri saree business and aims to support the complete customer journey:

1. Customer discovers sarees through the website or WhatsApp.
2. Customer can send a saree image and ask for visually similar products.
3. The AI sales agent searches the private product/inventory database and recommends matching sarees with prices through WhatsApp.
4. The customer can request a virtual try-on using their own photograph.
5. The visual agent runs a virtual try-on pipeline and returns the generated result through WhatsApp.
6. The customer can proceed to payment.
7. Inventory is reserved and sold using transactional backend logic.
8. An order and delivery record are created.

The project is simultaneously a research platform, production-style software project, and real-world business deployment.

---

## Core Research Areas

- Agentic AI and tool-using LLMs
- Multimodal product understanding
- Conversational recommendation systems
- Customer preference and memory
- Image similarity and multimodal retrieval
- Generative virtual try-on (VTON)
- Human/agent handoff and confidence-aware responses
- Secure transactional commerce workflows
- Evaluation of agent quality, recommendation quality, VTON quality, and business outcomes

---

## Phase 1 — Database Design

Phase 1 establishes the PostgreSQL data model that will support the rest of the system.

### Main domains

| Domain | Tables |
|---|---|
| Catalog | `products`, `product_images`, `product_prices` |
| Inventory | `inventory_items`, `inventory_transactions` |
| Customer + AI | `customers`, `customer_preferences`, `conversations`, `messages`, `recommendations` |
| Search | `product_embeddings` |
| VTON | `vton_jobs`, `generated_images` |
| Commerce | `carts`, `cart_items`, `orders`, `order_items`, `payments`, `deliveries` |
| Admin | `staff_users`, `audit_logs` |

### Important design decisions

- A **product** describes a saree design and metadata.
- An **inventory item** represents a physical saree/stock unit, allowing unique-piece tracking.
- Price is stored privately in `commerce.product_prices` and is not exposed by the public catalog view.
- Product images and customer/VTON images are stored in object storage; PostgreSQL stores references.
- Customer preferences keep a confidence score and provenance (`EXPLICIT`, `INFERRED`, etc.).
- Embeddings are prepared for multimodal similarity search using `pgvector`.
- Payment confirmation comes from a verified payment-provider webhook, not from the LLM.
- Inventory reservation will use database transactions and row-level locking.

---

## Repository Structure

```text
agentic-multimodal-ecommerce-vton-research/
├── README.md
├── docs/
│   ├── database/
│   │   ├── schema.sql
│   │   └── ERD.md
│   └── architecture/
├── src/
├── tests/
├── scripts/
├── notebooks/
└── deployment/
```

This repository will be built phase-by-phase so that each major capability is independently documented, tested, and committed.

---

## Security Principles

1. Never expose internal product prices through the public catalog API.
2. Keep customer photos in private object storage with short-lived signed URLs.
3. Never store card numbers, CVV, or other raw payment credentials.
4. Never give an LLM unrestricted SQL access.
5. Expose controlled backend tools such as `search_products`, `check_inventory`, `reserve_product`, and `get_order_status`.
6. Verify payment webhooks before changing payment/order state.
7. Use transactional inventory reservation to prevent double-selling unique sarees.
8. Add PostgreSQL Row-Level Security (RLS) once application roles are introduced.

---

## Roadmap

### Phase 1
Database design and PostgreSQL setup.

### Phase 2
Staff/admin inventory management and product catalog website.

### Phase 3
Multimodal product retrieval and recommendation engine.

### Phase 4
WhatsApp conversational sales agent, tool calling, and customer memory.

### Phase 5
Generative virtual try-on pipeline and evaluation.

### Phase 6
Payments, order management, delivery workflow, and production hardening.

### Phase 7
Research experiments, ablations, evaluation, real-customer deployment, and business analytics.

---

## Status

**Current phase: Phase 1 — Database Design**
