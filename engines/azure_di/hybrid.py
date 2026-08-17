"""Hybrid Azure DI: PP-DocLayoutV3 finds tables, ADI parses table crops only,
local PP-OCRv6 handles the rest of the page text.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image

from engines import BaseEngine, DocumentResult, OCRRegion, register_engine
from engines.azure_di.parse import parse_analyze_result
from engines.cost import azure_di_cost
from engines.pp_structure.parse import parse_ocr_regions

PDF_DPI = 300
TABLE_LABELS = {"table"}
# Padding around detected table boxes before crop (pixels at render DPI)
TABLE_PAD_PX = 8
# IoU above which a local OCR region is treated as inside a table and dropped
TABLE_OVERLAP_IOU = 0.3


def _pixmap_to_numpy(pix: pymupdf.Pixmap) -> np.ndarray:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def _normalize_bbox(raw: Any) -> list[float] | None:
    """Accept [x1,y1,x2,y2] or polygon points -> axis-aligned box."""
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None

    # Flat polygon: [x1,y1,x2,y2,...]
    if all(isinstance(v, (int, float)) for v in raw):
        if len(raw) == 4:
            x1, y1, x2, y2 = map(float, raw)
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        xs = [float(v) for v in raw[0::2]]
        ys = [float(v) for v in raw[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)]

    # Nested points: [[x,y], ...]
    if isinstance(raw[0], (list, tuple)):
        xs = [float(p[0]) for p in raw]
        ys = [float(p[1]) for p in raw]
        return [min(xs), min(ys), max(xs), max(ys)]

    return None


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _crop_table(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, int(bbox[0]) - TABLE_PAD_PX)
    y1 = max(0, int(bbox[1]) - TABLE_PAD_PX)
    x2 = min(w, int(bbox[2]) + TABLE_PAD_PX)
    y2 = min(h, int(bbox[3]) + TABLE_PAD_PX)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def _numpy_to_png_bytes(image: np.ndarray) -> bytes:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _extract_layout_boxes(layout_output: Any) -> list[dict[str, Any]]:
    """Parse PP-DocLayoutV3 predict() output into {label, bbox, score} dicts."""
    boxes: list[dict[str, Any]] = []
    items = layout_output if isinstance(layout_output, list) else [layout_output]

    for item in items:
        # Prefer to_dict / json-like structure
        data = item
        if hasattr(item, "json"):
            data = item.json
        elif hasattr(item, "to_dict"):
            data = item.to_dict()
        elif not isinstance(item, dict):
            data = {
                k: getattr(item, k)
                for k in ("boxes", "dt_boxes", "res", "label", "coordinate")
                if hasattr(item, k)
            }

        if isinstance(data, dict):
            candidates = (
                data.get("boxes")
                or data.get("dt_boxes")
                or data.get("res")
                or data.get("detections")
                or []
            )
            if isinstance(candidates, dict):
                candidates = [candidates]
        else:
            candidates = []

        for det in candidates:
            if not isinstance(det, dict):
                continue
            label = str(
                det.get("label")
                or det.get("cls_name")
                or det.get("class_name")
                or det.get("type")
                or ""
            ).lower()
            bbox = _normalize_bbox(
                det.get("coordinate")
                or det.get("bbox")
                or det.get("box")
                or det.get("poly")
            )
            if not bbox:
                continue
            score = float(det.get("score", det.get("confidence", 1.0)) or 1.0)
            boxes.append({"label": label, "bbox": bbox, "score": score})

    return boxes


@register_engine
class AzureHybridEngine(BaseEngine):
    """Local layout + OCR; Azure DI only for detected table crops."""

    name = "azure-hybrid"

    def __init__(self) -> None:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        from paddleocr import LayoutDetection, PaddleOCR

        endpoint = os.environ["AZURE_DI_ENDPOINT"]
        key = os.environ["AZURE_DI_KEY"]
        self._client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

        # PP-DocLayoutV3 for table detection
        try:
            self._layout = LayoutDetection(model_name="PP-DocLayoutV3")
        except TypeError:
            # Older paddleocr may not accept model_name kw; try positional / create_model
            from paddlex import create_model

            self._layout = create_model(model_name="PP-DocLayoutV3")

        self._ocr = PaddleOCR(ocr_version="PP-OCRv6")

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))

        render_s = layout_s = ocr_s = azure_s = 0.0
        azure_calls = 0
        tables_detected = 0
        pages_with_tables = 0

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

                # 1) Layout detection with PP-DocLayoutV3
                try:
                    t0 = time.perf_counter()
                    layout_raw = self._layout.predict(image, batch_size=1)
                    layout_s += time.perf_counter() - t0
                    layout_boxes = _extract_layout_boxes(layout_raw)
                except Exception as exc:
                    result.errors.append(
                        {"page": page_num, "stage": "layout", "error": str(exc)}
                    )
                    layout_boxes = []

                table_boxes = [
                    b for b in layout_boxes if b["label"] in TABLE_LABELS
                ]
                if table_boxes:
                    pages_with_tables += 1
                    tables_detected += len(table_boxes)

                # 2) Local OCR for page text (drop regions overlapping tables)
                try:
                    t0 = time.perf_counter()
                    raw_ocr = self._ocr.predict(image)
                    ocr_s += time.perf_counter() - t0
                    page_regions = parse_ocr_regions(raw_ocr, page_num)
                    if table_boxes:
                        page_regions = [
                            r
                            for r in page_regions
                            if not any(
                                _iou(r.bbox, tb["bbox"]) >= TABLE_OVERLAP_IOU
                                for tb in table_boxes
                            )
                        ]
                    result.regions.extend(page_regions)
                except Exception as exc:
                    result.errors.append(
                        {"page": page_num, "stage": "local_ocr", "error": str(exc)}
                    )

                # 3) Send each table crop to Azure DI prebuilt-layout
                for table_idx, tb in enumerate(table_boxes):
                    try:
                        from azure.ai.documentintelligence.models import (
                            AnalyzeDocumentRequest,
                        )

                        crop = _crop_table(image, tb["bbox"])
                        png_bytes = _numpy_to_png_bytes(crop)

                        t0 = time.perf_counter()
                        poller = self._client.begin_analyze_document(
                            "prebuilt-layout",
                            AnalyzeDocumentRequest(bytes_source=png_bytes),
                        )
                        analyze_result = poller.result()
                        azure_s += time.perf_counter() - t0
                        azure_calls += 1

                        _regions, tables = parse_analyze_result(analyze_result)
                        for table in tables:
                            # Remap page to source PDF page; keep crop-local coords in raw
                            table.page = page_num
                            table.raw = {
                                **table.raw,
                                "source_bbox": tb["bbox"],
                                "layout_score": tb["score"],
                                "table_index": table_idx,
                                "azure_call": azure_calls,
                            }
                            result.tables.append(table)

                        # If ADI returned no structured tables, keep OCR lines from crop as fallback text
                        if not tables and _regions:
                            for region in _regions:
                                offset_region = OCRRegion(
                                    page=page_num,
                                    text=region.text,
                                    bbox=[
                                        region.bbox[0] + tb["bbox"][0],
                                        region.bbox[1] + tb["bbox"][1],
                                        region.bbox[2] + tb["bbox"][0],
                                        region.bbox[3] + tb["bbox"][1],
                                    ],
                                    confidence=region.confidence,
                                )
                                result.regions.append(offset_region)
                    except Exception as exc:
                        result.errors.append(
                            {
                                "page": page_num,
                                "stage": "azure_table",
                                "table_index": table_idx,
                                "error": str(exc),
                            }
                        )

        result.timing = {
            "render_s": render_s,
            "layout_s": layout_s,
            "local_ocr_s": ocr_s,
            "azure_s": azure_s,
            "total_s": render_s + layout_s + ocr_s + azure_s,
        }
        result.cost = azure_di_cost(
            pages_billed=float(azure_calls),
            notes=(
                "Hybrid: billed only for table-crop ADI calls "
                "(PP-DocLayoutV3 detection + local PP-OCRv6 text)"
            ),
            breakdown={
                "azure_api_calls": azure_calls,
                "tables_detected": tables_detected,
                "pages_with_tables": pages_with_tables,
                "billing_unit": "1 ADI call per table crop (counts as 1 page)",
            },
        )
        result.metadata = {
            "mode": "hybrid",
            "layout_model": "PP-DocLayoutV3",
            "local_ocr": "PP-OCRv6",
            "azure_model": "prebuilt-layout",
            "tables_detected": tables_detected,
            "azure_api_calls": azure_calls,
        }
        return result
