"""API cost estimation helpers for billed OCR engines.

Prices are approximate list prices (USD) as of mid-2026 and can be overridden
via environment variables. Update when your contract/SKU differs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# Default list prices (USD)
# Azure DI Layout (S0): typically ~$10 / 1,000 pages for prebuilt models
# Mistral OCR 4: $4 / 1,000 pages ($2 / 1,000 in batch)
DEFAULT_AZURE_DI_USD_PER_1K_PAGES = 10.0
DEFAULT_MISTRAL_OCR_USD_PER_1K_PAGES = 4.0
DEFAULT_MISTRAL_OCR_BATCH_USD_PER_1K_PAGES = 2.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class CostEstimate:
    provider: str
    unit: str = "page"
    units_billed: float = 0.0
    usd_per_1k_units: float = 0.0
    usd_total: float = 0.0
    notes: str = ""
    breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "unit": self.unit,
            "units_billed": self.units_billed,
            "usd_per_1k_units": self.usd_per_1k_units,
            "usd_total": round(self.usd_total, 6),
            "notes": self.notes,
            "breakdown": self.breakdown,
        }


def azure_di_cost(
    pages_billed: float,
    *,
    notes: str = "",
    breakdown: dict[str, Any] | None = None,
) -> CostEstimate:
    rate = _env_float("AZURE_DI_USD_PER_1K_PAGES", DEFAULT_AZURE_DI_USD_PER_1K_PAGES)
    return CostEstimate(
        provider="azure-document-intelligence",
        unit="page",
        units_billed=pages_billed,
        usd_per_1k_units=rate,
        usd_total=(pages_billed / 1000.0) * rate,
        notes=notes
        or "prebuilt-layout (S0 list price approx; override AZURE_DI_USD_PER_1K_PAGES)",
        breakdown=breakdown or {},
    )


def mistral_ocr_cost(
    pages_billed: float,
    *,
    batch: bool = False,
    notes: str = "",
    breakdown: dict[str, Any] | None = None,
) -> CostEstimate:
    default = (
        DEFAULT_MISTRAL_OCR_BATCH_USD_PER_1K_PAGES
        if batch
        else DEFAULT_MISTRAL_OCR_USD_PER_1K_PAGES
    )
    rate = _env_float("MISTRAL_OCR_USD_PER_1K_PAGES", default)
    return CostEstimate(
        provider="mistral-ocr",
        unit="page",
        units_billed=pages_billed,
        usd_per_1k_units=rate,
        usd_total=(pages_billed / 1000.0) * rate,
        notes=notes
        or (
            "mistral-ocr-4 batch rate"
            if batch
            else "mistral-ocr-4 list price; override MISTRAL_OCR_USD_PER_1K_PAGES"
        ),
        breakdown=breakdown or {},
    )


def zero_cost(provider: str = "local") -> CostEstimate:
    return CostEstimate(
        provider=provider,
        unit="page",
        units_billed=0.0,
        usd_per_1k_units=0.0,
        usd_total=0.0,
        notes="Local / open-source engine — no API bill",
    )
