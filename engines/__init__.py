from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engines.cost import CostEstimate


@dataclass
class OCRRegion:
    page: int
    text: str
    bbox: list[float]
    confidence: float


@dataclass
class TableResult:
    page: int
    html: str
    headers: list[str]
    row_count: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    engine: str
    pdf_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    regions: list[OCRRegion] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    cost: CostEstimate | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "pdf_path": self.pdf_path,
            "metadata": self.metadata,
            "regions": [
                {
                    "page": r.page,
                    "text": r.text,
                    "bbox": r.bbox,
                    "confidence": r.confidence,
                }
                for r in self.regions
            ],
            "tables": [
                {
                    "page": t.page,
                    "html": t.html,
                    "headers": t.headers,
                    "row_count": t.row_count,
                    "raw": t.raw,
                }
                for t in self.tables
            ],
            "timing": self.timing,
            "cost": self.cost.to_dict() if self.cost else None,
            "errors": self.errors,
        }


class BaseEngine(ABC):
    name: str

    @abstractmethod
    def process(self, pdf_path: Path) -> DocumentResult:
        """Run OCR + table extraction on a single PDF and return structured results."""
        ...


# ---------------------------------------------------------------------------
# Engine registry -- engines register themselves on import
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseEngine]] = {}


def register_engine(cls: type[BaseEngine]) -> type[BaseEngine]:
    _REGISTRY[cls.name] = cls
    return cls


def get_engine_class(name: str) -> type[BaseEngine]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown engine '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def available_engines() -> list[str]:
    return sorted(_REGISTRY.keys())


def import_all_engines() -> None:
    """Import every engine sub-package so they self-register."""
    import importlib

    for module_name in (
        "engines.pp_structure",
        "engines.paddle_vl",
        "engines.azure_di",
        "engines.azure_di.hybrid",
        "engines.mistral_ocr",
        "engines.glm_ocr",
        "engines.olmocr",
        "engines.monkeyocr",
    ):
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass
