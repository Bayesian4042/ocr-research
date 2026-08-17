"""Normalize olmOCR pipeline output into shared dataclasses.

olmOCR writes results as Dolma-format JSONL in the workspace's `results/` dir,
and optionally markdown files when --markdown is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from engines import OCRRegion, TableResult


def parse_olmocr_output(
    workspace: Path,
) -> tuple[list[OCRRegion], list[TableResult]]:
    regions: list[OCRRegion] = []
    tables: list[TableResult] = []

    # olmOCR writes JSONL results under <workspace>/results/
    results_dir = workspace / "results"
    markdown_dir = workspace / "markdown"

    # Try JSONL results first (Dolma format)
    if results_dir.exists():
        for jsonl_file in sorted(results_dir.glob("*.jsonl")):
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = doc.get("text", "")
                page_num = doc.get("metadata", {}).get("page", 1)

                # Check for table markers in the text
                table_blocks = _extract_table_blocks(text)
                for tbl in table_blocks:
                    tables.append(tbl._replace(page=page_num) if hasattr(tbl, '_replace') else tbl)

                if text:
                    regions.append(
                        OCRRegion(
                            page=page_num,
                            text=text,
                            bbox=[0, 0, 0, 0],
                            confidence=1.0,
                        )
                    )

    # Fallback: read markdown files
    if not regions and markdown_dir.exists():
        for md_file in sorted(markdown_dir.glob("**/*.md")):
            text = md_file.read_text(encoding="utf-8")
            if text.strip():
                regions.append(OCRRegion(page=1, text=text, bbox=[0, 0, 0, 0], confidence=1.0))
                for tbl in _extract_table_blocks(text):
                    tables.append(tbl)

    return regions, tables


def _extract_table_blocks(text: str) -> list[TableResult]:
    """Best-effort extraction of markdown tables from olmOCR output."""
    tables: list[TableResult] = []

    # Match markdown tables (lines with | separators)
    md_table_pattern = re.compile(
        r"((?:^\|.*\|$\n?){2,})", re.MULTILINE
    )

    for match in md_table_pattern.finditer(text):
        table_text = match.group(1)
        rows = [r.strip() for r in table_text.strip().split("\n") if r.strip()]

        if len(rows) < 2:
            continue

        # Parse header row
        header_row = rows[0]
        headers = [c.strip() for c in header_row.split("|") if c.strip()]

        # Skip separator row (|---|---|)
        data_start = 1
        if len(rows) > 1 and re.match(r"^\|[\s\-:|]+\|$", rows[1]):
            data_start = 2

        html = _markdown_table_to_html(rows, data_start)
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


def _markdown_table_to_html(rows: list[str], data_start: int) -> str:
    parts = ["<table>"]

    # Header
    if rows:
        cells = [c.strip() for c in rows[0].split("|") if c.strip()]
        parts.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")

    # Data rows
    for row in rows[data_start:]:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    parts.append("</table>")
    return "".join(parts)
