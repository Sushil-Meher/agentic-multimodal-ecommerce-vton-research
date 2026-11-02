-- MTP Phase 1 seed data: 3 sample sarees with private prices and one
-- physical inventory piece each. Safe to run multiple times: every
-- insert conflicts on an existing unique key and does nothing on repeat.
-- Products start as DRAFT, so they stay out of the public catalog and the
-- agent's search until someone deliberately publishes them.

INSERT INTO catalog.products
(
    sku,
    name,
    description,
    category,
    fabric,
    primary_color,
    secondary_color,
    bandha_type,
    motif,
    thread_count,
    border_type,
    saree_length_cm,
    blouse_included,
    weaving_region,
    occasion,
    status,
    public_slug
)
VALUES
(
    'SAM-2026-0001',
    'Maroon Traditional Sambalpuri Saree',
    'Traditional Sambalpuri handloom saree with classic Bandha motifs.',
    'SAREE',
    'Cotton',
    'Maroon',
    'Black',
    'Traditional Bandha',
    'Shankha',
    120,
    'Wide',
    550,
    TRUE,
    'Sambalpur, Odisha',
    'Wedding',
    'DRAFT',
    'maroon-traditional-sambalpuri-saree'
),
(
    'SAM-2026-0002',
    'Black and Red Sambalpuri Saree',
    'Traditional Sambalpuri saree with geometric Bandha motifs.',
    'SAREE',
    'Cotton',
    'Black',
    'Red',
    'Traditional Bandha',
    'Geometric',
    100,
    'Medium',
    550,
    TRUE,
    'Sambalpur, Odisha',
    'Festival',
    'DRAFT',
    'black-red-sambalpuri-saree'
),
(
    'SAM-2026-0003',
    'Green Sambalpuri Saree',
    'Elegant green Sambalpuri handloom saree suitable for festive occasions.',
    'SAREE',
    'Cotton',
    'Dark Green',
    'Maroon',
    'Traditional Bandha',
    'Floral',
    80,
    'Medium',
    550,
    TRUE,
    'Sambalpur, Odisha',
    'Festival',
    'DRAFT',
    'green-sambalpuri-saree'
)
ON CONFLICT (sku) DO NOTHING;

-- Private prices (never exposed via catalog.public_products)
INSERT INTO commerce.product_prices (product_id, price, currency)
SELECT p.id, v.price, 'INR'
FROM (VALUES
    ('SAM-2026-0001', 4500.00),
    ('SAM-2026-0002', 3800.00),
    ('SAM-2026-0003', 2900.00)
) AS v(sku, price)
JOIN catalog.products p ON p.sku = v.sku
ON CONFLICT (product_id) WHERE is_current DO NOTHING;

-- One physical inventory piece per design (skip products that already
-- have an inventory item, whatever its code, not just an exact code match)
INSERT INTO commerce.inventory_items (product_id, inventory_code, status, location)
SELECT p.id, v.inventory_code, 'AVAILABLE', 'Sambalpur Warehouse'
FROM (VALUES
    ('SAM-2026-0001', 'SAM-2026-0001-INV-01'),
    ('SAM-2026-0002', 'SAM-2026-0002-INV-01'),
    ('SAM-2026-0003', 'SAM-2026-0003-INV-01')
) AS v(sku, inventory_code)
JOIN catalog.products p ON p.sku = v.sku
WHERE NOT EXISTS (
    SELECT 1 FROM commerce.inventory_items ii WHERE ii.product_id = p.id
)
ON CONFLICT (inventory_code) DO NOTHING;

-- =========================
-- VERIFICATION QUERIES
-- =========================

-- 1. Sample products
SELECT id, sku, name, primary_color, thread_count, saree_length_cm, public_slug
FROM catalog.products
WHERE sku LIKE 'SAM-2026-%';

-- 2. Public catalog view must never return a price column
SELECT * FROM catalog.public_products WHERE sku LIKE 'SAM-2026-%';

-- 3. Available sarees under 5000, joining product + current price + inventory
SELECT p.sku, p.name, pr.price, i.inventory_code, i.status
FROM catalog.products p
JOIN commerce.product_prices pr ON pr.product_id = p.id AND pr.is_current
JOIN commerce.inventory_items i ON i.product_id = p.id
WHERE pr.price < 5000
  AND i.status = 'AVAILABLE'
ORDER BY pr.price;
