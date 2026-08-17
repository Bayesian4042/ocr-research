"""PP-StructureV3 + PP-OCRv6 engine -- ported from the original main.py."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from engines import BaseEngine, DocumentResult, register_engine
from engines.pp_structure.parse import parse_ocr_regions, parse_table_results

PDF_DPI = 300


def _pixmap_to_numpy(pix: pymupdf.Pixmap) -> np.ndarray:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:
        img = np.repeat(img, 3, axis=2)
    return img


@register_engine
class PPStructureEngine(BaseEngine):
    name = "pp-structure"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR, TableRecognitionPipelineV2

        self._ocr = PaddleOCR(ocr_version="PP-OCRv6")
        try:
            self._table = TableRecognitionPipelineV2()
        except Exception as exc:
            raise RuntimeError(
                "TableRecognitionPipelineV2 deps missing. "
                'Install: uv add "paddlex[ocr]" && uv sync'
            ) from exc

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))
        render_s = ocr_s = table_s = 0.0

        with pymupdf.open(pdf_path) as doc:
            for page_idx in range(doc.page_count):
                page_num = page_idx + 1

                try:
                    t0 = time.perf_counter()
                    page = doc.load_page(page_idx)
                    mat = pymupdf.Matrix(PDF_DPI / 72.0, PDF_DPI / 72.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    image = _pixmap_to_numpy(pix)
                    render_s += time.perf_counter() - t0
                except Exception as exc:
                    result.errors.append(
                        {"page": page_num, "stage": "render", "error": str(exc)}
                    )
                    continue

                try:
                    t0 = time.perf_counter()
                    raw_ocr = self._ocr.predict(image)
                    ocr_s += time.perf_counter() - t0
                except Exception as exc:
                    result.errors.append(
                        {"page": page_num, "stage": "ocr", "error": str(exc)}
                    )
                    continue

                result.regions.extend(parse_ocr_regions(raw_ocr, page_num))

                try:
                    t0 = time.perf_counter()
                    raw_table = self._table.predict(image)
                    table_s += time.perf_counter() - t0
                    result.tables.extend(parse_table_results(raw_table, page_num))
                except Exception as exc:
                    result.errors.append(
                        {"page": page_num, "stage": "table", "error": str(exc)}
                    )

        result.timing = {
            "render_s": render_s,
            "ocr_s": ocr_s,
            "table_s": table_s,
        }
        return result
