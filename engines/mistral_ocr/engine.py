"""Mistral OCR 4 engine -- cloud API via mistralai SDK."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import pymupdf

from engines import BaseEngine, DocumentResult, register_engine
from engines.cost import mistral_ocr_cost
from engines.mistral_ocr.parse import parse_ocr_response


@register_engine
class MistralOCREngine(BaseEngine):
    name = "mistral"

    def __init__(self) -> None:
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral

        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self._batch = os.environ.get("MISTRAL_OCR_BATCH", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))

        try:
            with pymupdf.open(pdf_path) as doc:
                page_count = doc.page_count
        except Exception:
            page_count = 0

        t0 = time.perf_counter()
        try:
            pdf_bytes = pdf_path.read_bytes()
            b64 = base64.b64encode(pdf_bytes).decode("ascii")
            data_uri = f"data:application/pdf;base64,{b64}"

            ocr_response = self._client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": data_uri,
                },
                include_blocks=True,
                table_format="html",
            )
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "ocr", "error": str(exc)})
            result.timing = {"total_s": time.perf_counter() - t0}
            result.cost = mistral_ocr_cost(0.0, batch=self._batch, notes="API call failed")
            return result

        api_s = time.perf_counter() - t0

        pages = getattr(ocr_response, "pages", None) or []
        if pages:
            page_count = len(pages)

        try:
            regions, tables = parse_ocr_response(ocr_response)
            result.regions = regions
            result.tables = tables
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "parse", "error": str(exc)})

        result.timing = {"total_s": api_s}
        result.cost = mistral_ocr_cost(
            float(page_count),
            batch=self._batch,
            breakdown={"pdf_pages": page_count, "api_calls": 1},
        )
        return result
