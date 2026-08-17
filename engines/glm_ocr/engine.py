"""GLM-OCR engine -- Z.AI layout_parsing API."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests

from engines import BaseEngine, DocumentResult, register_engine
from engines.cost import CostEstimate
from engines.glm_ocr.parse import parse_layout_response

API_URL = "https://api.z.ai/api/paas/v4/layout_parsing"
MODEL = "glm-ocr"
# PDF limit: 50 MB / 30 pages per the API docs
MAX_PDF_BYTES = 50 * 1024 * 1024


@register_engine
class GlmOCREngine(BaseEngine):
    name = "glm-ocr"

    def __init__(self) -> None:
        self._api_key = os.environ.get("GLM_OCR_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "GLM_OCR_API_KEY env var is required for the glm-ocr engine"
            )

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))

        pdf_bytes = pdf_path.read_bytes()
        if len(pdf_bytes) > MAX_PDF_BYTES:
            result.errors.append(
                {
                    "page": 0,
                    "stage": "pre-check",
                    "error": f"PDF exceeds 50 MB limit ({len(pdf_bytes)} bytes)",
                }
            )
            return result

        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        suffix = pdf_path.suffix.lower().lstrip(".")
        mime = "application/pdf" if suffix == "pdf" else f"image/{suffix}"
        data_uri = f"data:{mime};base64,{b64}"

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "file": data_uri,
                },
                timeout=300,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.HTTPError as exc:
            error_text = exc.response.text if exc.response is not None else str(exc)
            result.errors.append(
                {
                    "page": 0,
                    "stage": "api",
                    "error": (
                        f"HTTP {exc.response.status_code}: {error_text}"
                        if exc.response
                        else str(exc)
                    ),
                }
            )
            result.timing = {"total_s": time.perf_counter() - t0}
            return result
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "api", "error": str(exc)})
            result.timing = {"total_s": time.perf_counter() - t0}
            return result

        api_s = time.perf_counter() - t0

        try:
            regions, tables = parse_layout_response(body)
            result.regions = regions
            result.tables = tables
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "parse", "error": str(exc)})

        # Extract page count from response
        data_info = body.get("data_info", {})
        page_count = data_info.get("num_pages", 0)
        if not page_count:
            layout_details = body.get("layout_details", [])
            page_count = len(layout_details) if layout_details else 1

        usage = body.get("usage", {})

        result.timing = {"total_s": api_s}
        result.cost = CostEstimate(
            provider="z-ai-glm-ocr",
            unit="page",
            units_billed=float(page_count),
            usd_per_1k_units=0.0,
            usd_total=0.0,
            notes="Z.AI GLM-OCR pricing TBD; set GLM_OCR_USD_PER_1K_PAGES if known",
            breakdown={
                "pdf_pages": page_count,
                "api_calls": 1,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )
        result.metadata = {
            "model": body.get("model", MODEL),
            "task_id": body.get("id"),
            "num_pages": page_count,
            "total_tokens": usage.get("total_tokens", 0),
        }
        return result
