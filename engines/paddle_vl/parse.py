"""Normalize PaddleOCR-VL 1.6 output into shared dataclasses."""

from __future__ import annotations

import re
from typing import Any

from engines import OCRRegion, TableResult


def parse_vl_result(
    page_data: dict[str, Any], page_number: int
) -> tuple[list[OCRRegion], list[TableResult]]:
    regions: list[OCRRegion] = []
    tables: list[TableResult] = []

    if not page_data:
        return regions, tables

    # PaddleOCR-VL outputs structured blocks with type labels
    blocks = page_data.get("blocks", page_data.get("res", []))
    if isinstance(blocks, dict):
        blocks = [blocks]

    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "text")
        text = block.get("text", block.get("content", ""))
        bbox = block.get("bbox", block.get("box", []))
        confidence = float(block.get("score", block.get("confidence", 1.0)))

        if isinstance(bbox, list) and len(bbox) >= 4:
            bbox_norm = bbox[:4] if len(bbox) == 4 else _flatten_bbox(bbox)
        else:
            bbox_norm = [0.0, 0.0, 0.0, 0.0]

        if block_type in ("table", "table_body"):
            html = block.get("html", block.get("pred_html", ""))
            if not html and text:
                html = f"<table><tr><td>{text}</td></tr></table>"
            headers = _extract_headers(html)
            row_count = max(html.count("<tr") - (1 if headers else 0), 0)
            tables.append(
                TableResult(
                    page=page_number,
                    html=html,
                    headers=headers,
                    row_count=row_count,
                    raw=block,
                )
            )
        elif text:
            regions.append(OCRRegion(page_number, str(text), bbox_norm, confidence))

    # Fallback: if no blocks, try markdown content
    if not blocks and "markdown" in page_data:
        md = page_data["markdown"]
        regions.append(OCRRegion(page_number, md, [0, 0, 0, 0], 1.0))

    return regions, tables


def _flatten_bbox(points: list) -> list[float]:
    """Convert polygon points [[x1,y1],[x2,y2],...] to [xmin,ymin,xmax,ymax]."""
    try:
        if isinstance(points[0], (list, tuple)):
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            return [min(xs), min(ys), max(xs), max(ys)]
    except (IndexError, TypeError, ValueError):
        pass
    return [float(x) for x in points[:4]]


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
