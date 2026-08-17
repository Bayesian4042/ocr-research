"""GLM-OCR engine -- Z.AI layout_parsing API."""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import requests

from engines import BaseEngine, DocumentResult, register_engine
from engines.cost import CostEstimate
from engines.glm_ocr.parse import parse_layout_response

API_URL = "https://api.z.ai/api/paas/v4/layout_parsing"
MODEL = "glm-ocr"
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PAGES_PER_CALL = 100


def _split_pdf(pdf_bytes: bytes, max_pages: int) -> list[bytes]:
    """Split a PDF into chunks of at most *max_pages* pages.

    Returns a list of PDF byte strings. If the PDF is already within limits,
    encrypted, or PyPDF is unavailable, returns the original bytes as a single chunk.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return [pdf_bytes]

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return [pdf_bytes]
        total = len(reader.pages)
    except Exception:
        return [pdf_bytes]

    if total <= max_pages:
        return [pdf_bytes]

    chunks: list[bytes] = []
    for start in range(0, total, max_pages):
        writer = PdfWriter()
        for page in reader.pages[start : start + max_pages]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    return chunks


@register_engine
class GlmOCREngine(BaseEngine):
    name = "glm-ocr"

    def __init__(self) -> None:
        self._api_key = os.environ.get("GLM_OCR_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "GLM_OCR_API_KEY env var is required for the glm-ocr engine"
            )

    def _call_api(self, pdf_bytes: bytes, pdf_path: Path) -> dict:
        """Send a single PDF chunk to the API and return the JSON body."""
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        suffix = pdf_path.suffix.lower().lstrip(".")
        mime = "application/pdf" if suffix == "pdf" else f"image/{suffix}"
        data_uri = f"data:{mime};base64,{b64}"

        # 10 min timeout -- large chunks can take a while
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "file": data_uri},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()

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

        chunks = _split_pdf(pdf_bytes, MAX_PAGES_PER_CALL)

        t0 = time.perf_counter()
        total_page_count = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        api_calls = 0
        page_offset = 0

        for chunk_idx, chunk_bytes in enumerate(chunks):
            try:
                body = self._call_api(chunk_bytes, pdf_path)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                try:
                    error_body = exc.response.json() if exc.response is not None else {}
                    error_text = (
                        error_body.get("error", {}).get("message", "")
                        or exc.response.text
                    )
                except Exception:
                    error_text = (
                        exc.response.text if exc.response is not None else str(exc)
                    )
                result.errors.append(
                    {
                        "page": page_offset,
                        "stage": "api",
                        "error": f"HTTP {status} (chunk {chunk_idx}): {error_text}",
                    }
                )
                continue
            except Exception as exc:
                result.errors.append(
                    {"page": page_offset, "stage": "api", "error": str(exc)}
                )
                continue

            api_calls += 1

            try:
                regions, tables = parse_layout_response(body, page_offset=page_offset)
                result.regions.extend(regions)
                result.tables.extend(tables)
            except Exception as exc:
                result.errors.append(
                    {"page": page_offset, "stage": "parse", "error": str(exc)}
                )

            data_info = body.get("data_info", {})
            chunk_pages = data_info.get("num_pages", 0)
            if not chunk_pages:
                layout_details = body.get("layout_details", [])
                chunk_pages = len(layout_details) if layout_details else 1
            total_page_count += chunk_pages
            page_offset += chunk_pages

            usage = body.get("usage", {})
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            if chunk_idx < len(chunks) - 1:
                time.sleep(2)

        api_s = time.perf_counter() - t0
        result.timing = {"total_s": api_s}
        result.cost = CostEstimate(
            provider="z-ai-glm-ocr",
            unit="page",
            units_billed=float(total_page_count),
            usd_per_1k_units=0.0,
            usd_total=0.0,
            notes="Z.AI GLM-OCR pricing TBD; set GLM_OCR_USD_PER_1K_PAGES if known",
            breakdown={
                "pdf_pages": total_page_count,
                "api_calls": api_calls,
                **total_usage,
            },
        )
        result.metadata = {
            "model": MODEL,
            "num_pages": total_page_count,
            "chunks": len(chunks),
            "total_tokens": total_usage["total_tokens"],
        }
        return result
