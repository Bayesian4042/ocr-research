"""Normalize MonkeyOCR v2 output into shared dataclasses.

MonkeyOCR outputs markdown files and optional JSON with layout/block info.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from engines import OCRRegion, TableResult


def parse_monkeyocr_output(
    output_dir: Path,
) -> tuple[list[OCRRegion], list[TableResult]]:
    regions: list[OCRRegion] = []
    tables: list[TableResult] = []

    # Try JSON output first
    for json_file in sorted(output_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        _parse_json_result(data, regions, tables)

    # Fall back to markdown output
    if not regions:
        for md_file in sorted(output_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            if text.strip():
                regions.append(OCRRegion(page=1, text=text, bbox=[0, 0, 0, 0], confidence=1.0))
                for tbl in _extract_markdown_tables(text):
                    tables.append(tbl)

    return regions, tables


def _parse_json_result(
    data: dict | list,
    regions: list[OCRRegion],
    tables: list[TableResult],
) -> None:
    items = data if isinstance(data, list) else data.get("blocks", data.get("pages", [data]))

    for item in items:
        if not isinstance(item, dict):
            continue

        page_num = item.get("page", item.get("page_number", 1))
        block_type = item.get("type", item.get("category", "text"))

        if block_type in ("table", "table_body"):
            html = item.get("html", item.get("content", ""))
            headers = _extract_html_headers(html)
            row_count = max(html.count("<tr") - (1 if headers else 0), 0)
            tables.append(
                TableResult(
                    page=page_num,
                    html=html,
                    headers=headers,
                    row_count=row_count,
                    raw=item,
                )
            )
        else:
            text = item.get("text", item.get("content", item.get("markdown", "")))
            bbox = item.get("bbox", item.get("box", [0, 0, 0, 0]))
            confidence = float(item.get("score", item.get("confidence", 1.0)))
            if text:
                regions.append(OCRRegion(page_num, str(text), bbox, confidence))

        # Recurse into children
        children = item.get("children", item.get("blocks", []))
        if children:
            _parse_json_result(children, regions, tables)


def _extract_markdown_tables(text: str) -> list[TableResult]:
    tables: list[TableResult] = []
    md_table_pattern = re.compile(r"((?:^\|.*\|$\n?){2,})", re.MULTILINE)

    for match in md_table_pattern.finditer(text):
        table_text = match.group(1)
        rows = [r.strip() for r in table_text.strip().split("\n") if r.strip()]

        if len(rows) < 2:
            continue

        headers = [c.strip() for c in rows[0].split("|") if c.strip()]
        data_start = 1
        if len(rows) > 1 and re.match(r"^\|[\s\-:|]+\|$", rows[1]):
            data_start = 2

        html = _md_to_html(rows, data_start)
        tables.append(
            TableResult(
                page=1,
                html=html,
                headers=headers,
                row_count=len(rows) - data_start,
                raw={"source": "markdown", "raw_text": table_text},
            )
        )

    return tables


def _md_to_html(rows: list[str], data_start: int) -> str:
    parts = ["<table>"]
    if rows:
        cells = [c.strip() for c in rows[0].split("|") if c.strip()]
        parts.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
    for row in rows[data_start:]:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    parts.append("</table>")
    return "".join(parts)


def _extract_html_headers(html: str) -> list[str]:
    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", html, re.DOTALL)
    if th_matches:
        return [re.sub(r"<[^>]+>", "", m).strip() for m in th_matches]

    first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    if first_tr:
        td_matches = re.findall(r"<td[^>]*>(.*?)</td>", first_tr.group(1), re.DOTALL)
        return [re.sub(r"<[^>]+>", "", m).strip() for m in td_matches]

    return []
