"""Normalize raw PaddleOCR / PP-StructureV3 output into shared dataclasses."""

from __future__ import annotations

from typing import Any

from engines import OCRRegion, TableResult


def _to_bbox(points: Any) -> list[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_ocr_regions(raw_ocr: Any, page_number: int) -> list[OCRRegion]:
    regions: list[OCRRegion] = []
    if not raw_ocr:
        return regions

    first = raw_ocr[0] if isinstance(raw_ocr, list) and raw_ocr else None

    # PaddleOCR 2.x: [[[points, (text, conf)], ...]]
    if isinstance(first, list):
        for entry in first:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            points, text_conf = entry[0], entry[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text = str(text_conf[0])
            conf = float(text_conf[1])
            regions.append(OCRRegion(page_number, text, _to_bbox(points), conf))
        return regions

    # PaddleOCR 3.x: [OCRResult(...)] with dt_polys / rec_texts / rec_scores
    items = raw_ocr if isinstance(raw_ocr, list) else [raw_ocr]
    for item in items:
        polys = _get_field(item, "dt_polys")
        texts = _get_field(item, "rec_texts")
        scores = _get_field(item, "rec_scores")
        if polys is None or texts is None:
            continue

        n = min(len(polys), len(texts))
        if scores is not None:
            n = min(n, len(scores))

        for i in range(n):
            pts = polys[i]
            if hasattr(pts, "tolist"):
                pts = pts.tolist()
            conf = float(scores[i]) if scores is not None else 0.0
            regions.append(OCRRegion(page_number, str(texts[i]), _to_bbox(pts), conf))

    return regions


def parse_table_results(raw_table: Any, page_number: int) -> list[TableResult]:
    if not raw_table:
        return []

    items = raw_table if isinstance(raw_table, list) else [raw_table]
    tables: list[TableResult] = []

    for item in items:
        if hasattr(item, "to_dict"):
            data = item.to_dict()
        elif isinstance(item, dict):
            data = item
        else:
            data = {
                k: getattr(item, k)
                for k in dir(item)
                if not k.startswith("_") and not callable(getattr(item, k, None))
            }

        html = str(data.get("html", data.get("pred_html", "")))
        headers = _extract_headers_from_html(html)
        row_count = html.count("<tr") - (1 if headers else 0)

        tables.append(
            TableResult(
                page=page_number,
                html=html,
                headers=headers,
                row_count=max(row_count, 0),
                raw=_to_jsonable(data),
            )
        )

    return tables


def _extract_headers_from_html(html: str) -> list[str]:
    """Best-effort header extraction from table HTML."""
    import re

    th_pattern = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL)
    matches = th_pattern.findall(html)
    if matches:
        return [re.sub(r"<[^>]+>", "", m).strip() for m in matches]

    # Fallback: first <tr> cells as headers
    first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    if first_tr:
        td_matches = re.findall(r"<td[^>]*>(.*?)</td>", first_tr.group(1), re.DOTALL)
        return [re.sub(r"<[^>]+>", "", m).strip() for m in td_matches]

    return []
