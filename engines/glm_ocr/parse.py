"""Normalize Z.AI GLM-OCR layout_parsing response into shared dataclasses."""

from __future__ import annotations

import re
from typing import Any

from engines import OCRRegion, TableResult


def parse_layout_response(
    body: dict[str, Any],
    page_offset: int = 0,
) -> tuple[list[OCRRegion], list[TableResult]]:
    regions: list[OCRRegion] = []
    tables: list[TableResult] = []

    # layout_details is a list of lists (one per page)
    layout_details = body.get("layout_details") or []

    for page_idx, page_blocks in enumerate(layout_details):
        page_num = page_offset + page_idx + 1
        if not isinstance(page_blocks, list):
            continue

        for block in page_blocks:
            if not isinstance(block, dict):
                continue

            label = block.get("label", "text")
            content = block.get("content", "")
            bbox_raw = block.get("bbox_2d", [0, 0, 0, 0])

            # bbox_2d is normalized 0-1; scale to pixel coords using page dims
            page_w = block.get("width", 1000)
            page_h = block.get("height", 1000)
            bbox = _scale_bbox(bbox_raw, page_w, page_h)

            if label == "table":
                html = _content_to_html(content)
                headers = _extract_headers(html)
                row_count = max(html.count("<tr") - (1 if headers else 0), 0)
                tables.append(
                    TableResult(
                        page=page_num,
                        html=html,
                        headers=headers,
                        row_count=row_count,
                        raw=block,
                    )
                )
            elif content:
                regions.append(
                    OCRRegion(
                        page=page_num,
                        text=str(content),
                        bbox=bbox,
                        confidence=1.0,
                    )
                )

    # Fallback: if no layout_details, use md_results
    if not regions and not tables:
        md = body.get("md_results", "")
        if md:
            regions.append(
                OCRRegion(page=page_offset + 1, text=md, bbox=[0, 0, 0, 0], confidence=1.0)
            )
            for tbl in _extract_md_tables(md):
                tables.append(tbl)

    return regions, tables


def _scale_bbox(bbox: list[float], page_w: int, page_h: int) -> list[float]:
    if len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(bbox[0]) * page_w,
        float(bbox[1]) * page_h,
        float(bbox[2]) * page_w,
        float(bbox[3]) * page_h,
    ]


def _content_to_html(content: str) -> str:
    """Convert table content (HTML or markdown) to HTML."""
    if "<table" in content.lower():
        return content

    # Try converting markdown table to HTML
    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    if len(lines) >= 2 and "|" in lines[0]:
        return _md_table_to_html(lines)

    return f"<table><tr><td>{content}</td></tr></table>"


def _md_table_to_html(rows: list[str]) -> str:
    parts = ["<table>"]

    if rows:
        cells = [c.strip() for c in rows[0].split("|") if c.strip()]
        parts.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")

    data_start = 1
    if len(rows) > 1 and re.match(r"^\|[\s\-:|]+\|$", rows[1]):
        data_start = 2

    for row in rows[data_start:]:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    parts.append("</table>")
    return "".join(parts)


def _extract_headers(html: str) -> list[str]:
    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", html, re.DOTALL)
    if th_matches:
        return [re.sub(r"<[^>]+>", "", m).strip() for m in th_matches]

    first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    if first_tr:
        td_matches = re.findall(r"<td[^>]*>(.*?)</td>", first_tr.group(1), re.DOTALL)
        return [re.sub(r"<[^>]+>", "", m).strip() for m in td_matches]

    return []


def _extract_md_tables(text: str) -> list[TableResult]:
    tables: list[TableResult] = []
    pattern = re.compile(r"((?:^\|.*\|$\n?){2,})", re.MULTILINE)

    for match in pattern.finditer(text):
        table_text = match.group(1)
        rows = [r.strip() for r in table_text.strip().split("\n") if r.strip()]
        if len(rows) < 2:
            continue

        headers = [c.strip() for c in rows[0].split("|") if c.strip()]
        data_start = 1
        if len(rows) > 1 and re.match(r"^\|[\s\-:|]+\|$", rows[1]):
            data_start = 2

        html = _md_table_to_html(rows)
        tables.append(
            TableResult(
                page=1,
                html=html,
                headers=headers,
                row_count=len(rows) - data_start,
                raw={"source": "markdown_fallback", "raw_text": table_text},
            )
        )

    return tables
