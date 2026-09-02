from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Product(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sku: str
    name: str
    description: str | None
    category: str
    fabric: str | None
    primary_color: str | None
    secondary_color: str | None
    bandha_type: str | None
    motif: str | None
    thread_count: int | None
    border_type: str | None
    saree_length_cm: float | None
    blouse_included: bool
    weaving_region: str | None
    occasion: str | None
    weight_grams: float | None
    public_slug: str
    status: str
    created_at: datetime
    updated_at: datetime
