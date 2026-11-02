-- Dedicated ACTIVE catalog for the evaluation harness.
-- The real SAM-2026-xxxx sample products stay DRAFT and untouched; a
-- benchmark should not depend on mutable production/business data anyway,
-- so this is its own small, stable fixture set. Safe to re-run (same
-- ON CONFLICT pattern as docs/database/seed.sql).

INSERT INTO catalog.products
(
    sku, name, description, category, fabric, primary_color, secondary_color,
    bandha_type, motif, thread_count, border_type, saree_length_cm,
    blouse_included, weaving_region, occasion, status, public_slug
)
VALUES
(
    'EVAL-0001', 'Maroon Wedding Bandha Saree',
    'Cotton Sambalpuri saree with Traditional Bandha weave and Shankha motif.',
    'SAREE', 'Cotton', 'Maroon', 'Black', 'Traditional Bandha', 'Shankha',
    120, 'Wide', 550, TRUE, 'Sambalpur, Odisha', 'Wedding', 'ACTIVE', 'eval-maroon-wedding-bandha'
),
(
    'EVAL-0002', 'Black Festival Geometric Saree',
    'Cotton Sambalpuri saree with Traditional Bandha weave and Geometric motif.',
    'SAREE', 'Cotton', 'Black', 'Red', 'Traditional Bandha', 'Geometric',
    100, 'Medium', 550, TRUE, 'Sambalpur, Odisha', 'Festival', 'ACTIVE', 'eval-black-festival-geometric'
),
(
    'EVAL-0003', 'Green Festival Floral Saree',
    'Cotton Sambalpuri saree with Traditional Bandha weave and Floral motif.',
    'SAREE', 'Cotton', 'Dark Green', 'Maroon', 'Traditional Bandha', 'Floral',
    80, 'Medium', 550, TRUE, 'Sambalpur, Odisha', 'Festival', 'ACTIVE', 'eval-green-festival-floral'
),
(
    'EVAL-0004', 'Ivory Silk Pasapalli Wedding Saree',
    'Silk Sambalpuri saree with Pasapalli Bandha checkerboard weave and Chakra motif.',
    'SAREE', 'Silk', 'Ivory', 'Gold', 'Pasapalli Bandha', 'Chakra',
    140, 'Wide', 560, TRUE, 'Sambalpur, Odisha', 'Wedding', 'ACTIVE', 'eval-ivory-silk-pasapalli-wedding'
),
(
    'EVAL-0005', 'Yellow Festival Bichitrapuri Saree',
    'Cotton Sambalpuri saree with Bichitrapuri Bandha weave and Phula motif.',
    'SAREE', 'Cotton', 'Yellow', 'Maroon', 'Bichitrapuri Bandha', 'Phula',
    100, 'Medium', 550, TRUE, 'Sambalpur, Odisha', 'Festival', 'ACTIVE', 'eval-yellow-festival-bichitrapuri'
),
(
    'EVAL-0006', 'Blue Casual Machha Saree',
    'Lightweight Cotton Sambalpuri saree with Traditional Bandha weave and Machha motif.',
    'SAREE', 'Cotton', 'Blue', 'White', 'Traditional Bandha', 'Machha',
    90, 'Narrow', 540, FALSE, 'Sambalpur, Odisha', 'Casual', 'ACTIVE', 'eval-blue-casual-machha'
),
(
    'EVAL-0007', 'Maroon Silk Rudraksha Wedding Saree',
    'Premium Silk Sambalpuri saree with Rudraksha Bandha weave and Shankha motif.',
    'SAREE', 'Silk', 'Maroon', 'Gold', 'Rudraksha Bandha', 'Shankha',
    120, 'Wide', 560, TRUE, 'Sambalpur, Odisha', 'Wedding', 'ACTIVE', 'eval-maroon-silk-rudraksha-wedding'
),
(
    'EVAL-0008', 'Pink Festival Geometric Saree',
    'Cotton Sambalpuri saree with Traditional Bandha weave and Geometric motif.',
    'SAREE', 'Cotton', 'Pink', 'Black', 'Traditional Bandha', 'Geometric',
    100, 'Medium', 550, TRUE, 'Sambalpur, Odisha', 'Festival', 'ACTIVE', 'eval-pink-festival-geometric'
)
ON CONFLICT (sku) DO NOTHING;

INSERT INTO commerce.product_prices (product_id, price, currency)
SELECT p.id, v.price, 'INR'
FROM (VALUES
    ('EVAL-0001', 4500.00),
    ('EVAL-0002', 5200.00),
    ('EVAL-0003', 3800.00),
    ('EVAL-0004', 9500.00),
    ('EVAL-0005', 4200.00),
    ('EVAL-0006', 2600.00),
    ('EVAL-0007', 12000.00),
    ('EVAL-0008', 3200.00)
) AS v(sku, price)
JOIN catalog.products p ON p.sku = v.sku
ON CONFLICT (product_id) WHERE is_current DO NOTHING;

-- most fixtures are AVAILABLE; two are deliberately not, to give
-- availability-related benchmark queries something real to distinguish
INSERT INTO commerce.inventory_items (product_id, inventory_code, status, location)
SELECT p.id, v.inventory_code, v.status::commerce.inventory_status, 'Sambalpur Warehouse'
FROM (VALUES
    ('EVAL-0001', 'EVAL-0001-INV-01', 'AVAILABLE'),
    ('EVAL-0002', 'EVAL-0002-INV-01', 'AVAILABLE'),
    ('EVAL-0003', 'EVAL-0003-INV-01', 'AVAILABLE'),
    ('EVAL-0004', 'EVAL-0004-INV-01', 'AVAILABLE'),
    ('EVAL-0005', 'EVAL-0005-INV-01', 'AVAILABLE'),
    ('EVAL-0006', 'EVAL-0006-INV-01', 'AVAILABLE'),
    ('EVAL-0007', 'EVAL-0007-INV-01', 'SOLD'),
    ('EVAL-0008', 'EVAL-0008-INV-01', 'RESERVED')
) AS v(sku, inventory_code, status)
JOIN catalog.products p ON p.sku = v.sku
WHERE NOT EXISTS (
    SELECT 1 FROM commerce.inventory_items ii WHERE ii.product_id = p.id
)
ON CONFLICT (inventory_code) DO NOTHING;

-- verification
SELECT p.sku, p.status, p.primary_color, p.fabric, p.bandha_type, p.thread_count,
       p.occasion, pr.price, i.status AS inventory_status
FROM catalog.products p
JOIN commerce.product_prices pr ON pr.product_id = p.id AND pr.is_current
JOIN commerce.inventory_items i ON i.product_id = p.id
WHERE p.sku LIKE 'EVAL-%'
ORDER BY p.sku;
