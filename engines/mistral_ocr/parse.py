"""Normalize Mistral OCR 4 response into shared dataclasses."""

from __future__ import annotations

import re
from typing import Any

from engines import OCRRegion, TableResult


def parse_ocr_response(
    response: Any,
) -> tuple[list[OCRRegion], list[TableResult]]:
    regions: list[OCRRegion] = []
    tables: list[TableResult] = []

    pages = getattr(response, "pages", None) or []

    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1

        # Extract text blocks with bounding boxes
        blocks = getattr(page, "blocks", None) or []
        for block in blocks:
            block_type = getattr(block, "type", "text")
            text = getattr(block, "text", getattr(block, "content", ""))
            bbox_raw = getattr(block, "bbox", None)
            confidence = float(getattr(block, "confidence", 1.0) or 1.0)

            bbox = _normalize_bbox(bbox_raw) if bbox_raw else [0, 0, 0, 0]

            if block_type == "table":
                html = getattr(block, "html", "") or text
                headers = _extract_headers(html)
                row_count = max(html.count("<tr") - (1 if headers else 0), 0)
                tables.append(
                    TableResult(
                        page=page_num,
                        html=html,
                        headers=headers,
                        row_count=row_count,
                        raw=_block_to_dict(block),
                    )
                )
            elif text:
                regions.append(OCRRegion(page_num, str(text), bbox, confidence))

        # Fallback: page-level markdown if no blocks
        if not blocks:
            markdown = getattr(page, "markdown", "")
            if markdown:
                regions.append(OCRRegion(page_num, markdown, [0, 0, 0, 0], 1.0))

    return regions, tables


def _normalize_bbox(bbox: Any) -> list[float]:
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(x) for x in bbox[:4]]
    if hasattr(bbox, "x") and hasattr(bbox, "y"):
        return [
            float(bbox.x),
            float(bbox.y),
            float(getattr(bbox, "x2", bbox.x)),
            float(getattr(bbox, "y2", bbox.y)),
        ]
    return [0, 0, 0, 0]


def _block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    result = {}
    for attr in ("type", "text", "bbox", "confidence", "html"):
        val = getattr(block, attr, None)
        if val is not None:
            result[attr] = str(val) if not isinstance(val, (str, int, float, bool)) else val
    return result


def _extract_headers(html: str) -> list[str]:
    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", html, re.DOTALL)
    if th_matches:
        return [re.sub(r"<[^>]+>", "", m).strip() for m in th_matches]

    first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    if first_tr:
        td_matches = re.findall(
            r"<td[^>]*>(.*?)</td>", first_tr.group(1), re.DOTALL
        )
        return [re.sub(r"<[^>]+>", "", m).strip() for m in td_matches]

    return []
