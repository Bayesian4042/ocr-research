"""Azure Document Intelligence prebuilt-layout engine (full-document)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pymupdf

from engines import BaseEngine, DocumentResult, register_engine
from engines.azure_di.parse import parse_analyze_result
from engines.cost import azure_di_cost


@register_engine
class AzureDIEngine(BaseEngine):
    name = "azure"

    def __init__(self) -> None:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        endpoint = os.environ["AZURE_DI_ENDPOINT"]
        key = os.environ["AZURE_DI_KEY"]

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
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
            with open(pdf_path, "rb") as f:
                poller = self._client.begin_analyze_document(
                    "prebuilt-layout",
                    body=f,
                )
                analyze_result = poller.result()
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "analyze", "error": str(exc)})
            result.timing = {"total_s": time.perf_counter() - t0}
            result.cost = azure_di_cost(0.0, notes="Failed before/during analysis")
            return result

        analyze_s = time.perf_counter() - t0

        # Prefer page count from the analyze result when available
        if hasattr(analyze_result, "pages") and analyze_result.pages:
            page_count = len(analyze_result.pages)

        try:
            regions, tables = parse_analyze_result(analyze_result)
            result.regions = regions
            result.tables = tables
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "parse", "error": str(exc)})

        result.timing = {"total_s": analyze_s}
        result.cost = azure_di_cost(
            float(page_count),
            notes="Full-document prebuilt-layout: 1 billable page per PDF page",
            breakdown={"pdf_pages": page_count, "azure_api_calls": 1},
        )
        return result
