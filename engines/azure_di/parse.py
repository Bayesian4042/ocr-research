"""Normalize Azure Document Intelligence AnalyzeResult into shared dataclasses."""

from __future__ import annotations

from typing import Any

from engines import OCRRegion, TableResult


def parse_analyze_result(
    result: Any,
) -> tuple[list[OCRRegion], list[TableResult]]:
    regions: list[OCRRegion] = []
    tables: list[TableResult] = []

    # Extract text regions from pages
    if hasattr(result, "pages") and result.pages:
        for page in result.pages:
            page_num = page.page_number

            if page.lines:
                for line in page.lines:
                    bbox = _polygon_to_bbox(line.polygon) if line.polygon else [0, 0, 0, 0]
                    confidence = getattr(line, "confidence", 1.0) or 1.0
                    regions.append(
                        OCRRegion(
                            page=page_num,
                            text=line.content,
                            bbox=bbox,
                            confidence=float(confidence),
                        )
                    )

    # Extract tables
    if hasattr(result, "tables") and result.tables:
        for table in result.tables:
            page_num = _table_page_number(table)
            html = _table_to_html(table)
            headers = _extract_table_headers(table)
            row_count = table.row_count - 1 if headers else table.row_count

            tables.append(
                TableResult(
                    page=page_num,
                    html=html,
                    headers=headers,
                    row_count=max(row_count, 0),
                    raw={
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cell_count": len(table.cells) if table.cells else 0,
                    },
                )
            )

    return regions, tables


def _polygon_to_bbox(polygon: list[float]) -> list[float]:
    """Azure polygons are flat [x1,y1,x2,y2,...,x4,y4]."""
    if not polygon:
        return [0, 0, 0, 0]
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _table_page_number(table: Any) -> int:
    if table.bounding_regions:
        return table.bounding_regions[0].page_number
    if table.cells:
        for cell in table.cells:
            if cell.bounding_regions:
                return cell.bounding_regions[0].page_number
    return 1


def _table_to_html(table: Any) -> str:
    if not table.cells:
        return ""

    grid: dict[tuple[int, int], str] = {}
    for cell in table.cells:
        grid[(cell.row_index, cell.column_index)] = cell.content

    rows = []
    for r in range(table.row_count):
        cells = []
        for c in range(table.column_count):
            content = grid.get((r, c), "")
            tag = "th" if r == 0 else "td"
            cells.append(f"<{tag}>{content}</{tag}>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return "<table>" + "".join(rows) + "</table>"


def _extract_table_headers(table: Any) -> list[str]:
    if not table.cells:
        return []

    headers = []
    for cell in table.cells:
        if cell.row_index == 0:
            headers.append(cell.content)

    return headers
