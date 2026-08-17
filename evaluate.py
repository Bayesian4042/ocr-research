"""Evaluate OCR engine outputs against ground truth annotations.

Metadata GT comes from CSV files in data/gt/ (carrier, SCAC, mode, dates).
Optional per-PDF JSON under data/test-dataset/ground-truth/ can add table GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_DIR = Path("outputs/benchmark")
DEFAULT_GT_CSV_DIR = Path("data/gt")
DEFAULT_GROUND_TRUTH_DIR = Path("data/test-dataset/ground-truth")
CATEGORIES = ("orientation", "scanned", "complex-tables")

# CSV column -> metadata field used for scoring
CSV_FIELD_MAP = {
    "carrier_inferred": "carrier_name",
    "scac_inferred": "scac",
    "mode_inferred": "mode",
    "effective_date": "effective_date",
    "end_date": "end_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR benchmark results against ground truth"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory with benchmark results",
    )
    parser.add_argument(
        "--gt-csv-dir",
        type=Path,
        default=DEFAULT_GT_CSV_DIR,
        help="Directory with metadata ground-truth CSV files",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Optional directory with per-PDF JSON (tables, extras)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help="Comma-separated engine names to evaluate (default: all found)",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    # Strip known file extensions explicitly (Path.stem mis-handles dates like "5.1.2023")
    stem = name
    for ext in (
        ".pdf",
        ".xlsx",
        ".xls",
        ".csv",
        ".docx",
        ".doc",
        ".png",
        ".jpg",
        ".tiff",
    ):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def load_gt_from_csv(csv_dir: Path) -> dict[str, dict[str, Any]]:
    """Load metadata ground truth from CSV files in data/gt/.

    Expected columns:
      file_name, file_type, carrier_inferred, scac_inferred,
      mode_inferred, effective_date, end_date
    """
    gt_data: dict[str, dict[str, Any]] = {}
    if not csv_dir.exists():
        return gt_data

    for csv_file in sorted(csv_dir.glob("*.csv")):
        with csv_file.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = (row.get("file_name") or "").strip()
                if not file_name:
                    continue

                metadata = {
                    target: _empty_to_none(row.get(source))
                    for source, target in CSV_FIELD_MAP.items()
                }

                key = sanitize_filename(file_name)
                gt_data[key] = {
                    "source_pdf": file_name,
                    "file_type": _empty_to_none(row.get("file_type")),
                    "metadata": metadata,
                    "tables": [],
                }

    return gt_data


def load_gt_from_json(gt_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional per-PDF JSON ground truth (mainly for tables)."""
    gt_data: dict[str, dict[str, Any]] = {}
    if not gt_dir.exists():
        return gt_data

    for gt_file in gt_dir.glob("*.json"):
        try:
            data = json.loads(gt_file.read_text(encoding="utf-8"))
            gt_data[gt_file.stem] = data
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    return gt_data


def load_ground_truth(
    csv_dir: Path,
    json_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge CSV metadata GT with optional JSON table GT."""
    gt_data = load_gt_from_csv(csv_dir)

    if json_dir is not None:
        for key, json_gt in load_gt_from_json(json_dir).items():
            if key in gt_data:
                # Prefer CSV metadata; keep JSON tables if present
                if json_gt.get("tables"):
                    gt_data[key]["tables"] = json_gt["tables"]
                if json_gt.get("category"):
                    gt_data[key]["category"] = json_gt["category"]
            else:
                gt_data[key] = json_gt

    return gt_data


def find_engine_results(benchmark_dir: Path, engine: str) -> dict[str, dict[str, Path]]:
    """Return {category: {doc_stem: result_json_path}} for an engine."""
    engine_dir = benchmark_dir / engine
    results: dict[str, dict[str, Path]] = {}

    if not engine_dir.exists():
        return results

    for cat in CATEGORIES:
        cat_dir = engine_dir / cat
        if not cat_dir.exists():
            continue

        doc_results: dict[str, Path] = {}
        for doc_dir in sorted(cat_dir.iterdir()):
            result_file = doc_dir / "result.json"
            if result_file.exists():
                doc_results[doc_dir.name] = result_file

        if doc_results:
            results[cat] = doc_results

    return results


def score_metadata(
    engine_result: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Compare extracted metadata fields against ground truth."""
    gt_meta = ground_truth.get("metadata", {})
    engine_meta = engine_result.get("metadata", {})

    scores: dict[str, Any] = {
        "fields": {},
        "exact_matches": 0,
        "fuzzy_matches": 0,
        "total_fields": 0,
        "missing_gt": 0,
    }

    for field, gt_value in gt_meta.items():
        scores["total_fields"] += 1

        if gt_value is None:
            scores["missing_gt"] += 1
            scores["fields"][field] = {"status": "no_ground_truth"}
            continue

        engine_value = engine_meta.get(field)
        source_text = None
        if engine_value is None:
            search_result = _search_in_regions(engine_result, str(gt_value), field)
            if search_result is not None:
                engine_value, source_text = search_result

        if engine_value is not None:
            exact = _normalize(str(engine_value)) == _normalize(str(gt_value))
            fuzzy = _fuzzy_score(str(engine_value), str(gt_value))

            if exact:
                scores["exact_matches"] += 1
            if fuzzy >= 0.8:
                scores["fuzzy_matches"] += 1

            scores["fields"][field] = {
                "gt": gt_value,
                "extracted": engine_value,
                "source_text": source_text,
                "exact_match": exact,
                "fuzzy_score": fuzzy,
            }
        else:
            scores["fields"][field] = {
                "gt": gt_value,
                "extracted": None,
                "source_text": None,
                "exact_match": False,
                "fuzzy_score": 0.0,
            }

    scorable = scores["total_fields"] - scores["missing_gt"]
    scores["exact_match_rate"] = (
        scores["exact_matches"] / scorable if scorable > 0 else None
    )
    scores["fuzzy_match_rate"] = (
        scores["fuzzy_matches"] / scorable if scorable > 0 else None
    )

    return scores


def score_tables(
    engine_result: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Compare extracted tables against ground truth tables."""
    gt_tables = ground_truth.get("tables", [])
    engine_tables = engine_result.get("tables", [])

    scores: dict[str, Any] = {
        "gt_table_count": len(gt_tables),
        "extracted_table_count": len(engine_tables),
        "table_scores": [],
    }

    for gt_idx, gt_table in enumerate(gt_tables):
        gt_page = gt_table.get("page")
        gt_headers = gt_table.get("headers", [])
        gt_row_count = gt_table.get("row_count")

        if gt_page is None and not gt_headers and gt_row_count is None:
            scores["table_scores"].append({"status": "no_ground_truth"})
            continue

        # Find best matching engine table (by page, then header overlap)
        best_match = _find_best_table_match(engine_tables, gt_table)

        if best_match is None:
            scores["table_scores"].append(
                {
                    "gt_page": gt_page,
                    "gt_headers": gt_headers,
                    "status": "not_found",
                    "header_accuracy": 0.0,
                    "row_count_accuracy": 0.0,
                }
            )
            continue

        # Score headers
        ext_headers = best_match.get("headers", [])
        header_accuracy = (
            _header_overlap(gt_headers, ext_headers) if gt_headers else None
        )

        # Score row count
        ext_row_count = best_match.get("row_count", 0)
        if gt_row_count is not None and gt_row_count > 0:
            row_accuracy = 1.0 - abs(ext_row_count - gt_row_count) / gt_row_count
            row_accuracy = max(row_accuracy, 0.0)
        else:
            row_accuracy = None

        # Score sample rows text similarity
        gt_samples = gt_table.get("sample_rows", [])
        cell_similarity = (
            _score_sample_rows(gt_samples, best_match) if gt_samples else None
        )

        scores["table_scores"].append(
            {
                "gt_page": gt_page,
                "gt_headers": gt_headers,
                "ext_headers": ext_headers,
                "header_accuracy": header_accuracy,
                "gt_row_count": gt_row_count,
                "ext_row_count": ext_row_count,
                "row_count_accuracy": row_accuracy,
                "cell_text_similarity": cell_similarity,
            }
        )

    # Aggregate
    valid_header_scores = [
        s["header_accuracy"]
        for s in scores["table_scores"]
        if s.get("header_accuracy") is not None
    ]
    valid_row_scores = [
        s["row_count_accuracy"]
        for s in scores["table_scores"]
        if s.get("row_count_accuracy") is not None
    ]

    scores["avg_header_accuracy"] = (
        sum(valid_header_scores) / len(valid_header_scores)
        if valid_header_scores
        else None
    )
    scores["avg_row_count_accuracy"] = (
        sum(valid_row_scores) / len(valid_row_scores) if valid_row_scores else None
    )

    return scores


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# Date formats to try when parsing GT and OCR dates
_DATE_FORMATS = [
    "%m.%d.%Y",  # 11.29.2022
    "%m/%d/%Y",  # 11/29/2022
    "%m-%d-%Y",  # 11-29-2022
    "%m.%d.%y",  # 11.29.22
    "%m/%d/%y",  # 11/29/22
    "%B %d, %Y",  # November 29, 2022
    "%b %d, %Y",  # Nov 29, 2022
    "%b. %d, %Y",  # Nov. 29, 2022
    "%d %B %Y",  # 29 November 2022
    "%d %b %Y",  # 29 Nov 2022
    "%Y-%m-%d",  # 2022-11-29
    "%Y/%m/%d",  # 2022/11/29
    "%m.%d.%Y",  # 04.15.2023
]

_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_date(text: str) -> datetime | None:
    """Try to parse a date string in various formats."""
    cleaned = text.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _extract_dates_from_text(text: str) -> list[datetime]:
    """Extract all recognizable dates from a block of text."""
    dates: list[datetime] = []

    # Numeric patterns: M/D/YYYY, M.D.YYYY, M-D-YYYY, YYYY-MM-DD
    for m in re.finditer(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b", text):
        parsed = _parse_date(m.group(0))
        if parsed:
            dates.append(parsed)

    for m in re.finditer(r"\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b", text):
        parsed = _parse_date(m.group(0))
        if parsed:
            dates.append(parsed)

    # Named months: "November 29, 2022", "Nov 29, 2022", "29 November 2022"
    month_pat = "|".join(_MONTH_NAMES.keys())
    for m in re.finditer(
        rf"\b({month_pat})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", text, re.IGNORECASE
    ):
        parsed = _parse_date(m.group(0))
        if parsed:
            dates.append(parsed)

    for m in re.finditer(
        rf"\b(\d{{1,2}})\s+({month_pat})\.?\s+(\d{{4}})\b", text, re.IGNORECASE
    ):
        parsed = _parse_date(m.group(0))
        if parsed:
            dates.append(parsed)

    return dates


def _is_date_field(field_name: str) -> bool:
    return "date" in field_name.lower()


def _fuzzy_score(a: str, b: str) -> float:
    """Normalized similarity between two strings."""
    try:
        from rapidfuzz import fuzz

        return fuzz.ratio(_normalize(a), _normalize(b)) / 100.0
    except ImportError:
        na, nb = _normalize(a), _normalize(b)
        if na == nb:
            return 1.0
        if na in nb or nb in na:
            return 0.8
        return 0.0


def _search_date_in_regions(
    result: dict[str, Any], gt_date_str: str
) -> tuple[str, str] | None:
    """Search for a date value in OCR regions using date normalization.

    Returns (matched_value, source_region_snippet) or None.
    """
    gt_date = _parse_date(gt_date_str)
    if gt_date is None:
        return None

    regions = result.get("regions", [])
    for region in regions:
        text = region.get("text", "")
        found_dates = _extract_dates_from_text(text)
        for d in found_dates:
            if d.date() == gt_date.date():
                return gt_date_str, text[:200]
    return None


def _search_in_regions(
    result: dict[str, Any], target: str, field_name: str = ""
) -> tuple[str, str] | None:
    """Search for a target string in OCR regions.

    For date fields, uses date normalization to match across formats.
    Returns (matched_value, source_region_snippet) or None.
    """
    if _is_date_field(field_name):
        date_match = _search_date_in_regions(result, target)
        if date_match is not None:
            return date_match

    target_norm = _normalize(target)
    regions = result.get("regions", [])

    best_match_text: str | None = None
    best_score = 0.0

    for region in regions:
        text = region.get("text", "")
        text_norm = _normalize(text)

        if target_norm == text_norm:
            return target, text[:200]

        if target_norm in text_norm:
            return target, text[:200]

        score = _fuzzy_score(target, text)
        if score > best_score:
            best_score = score
            best_match_text = text

    if best_score >= 0.6 and best_match_text is not None:
        return best_match_text, best_match_text[:200]

    return None


def _find_best_table_match(
    engine_tables: list[dict[str, Any]],
    gt_table: dict[str, Any],
) -> dict[str, Any] | None:
    if not engine_tables:
        return None

    gt_page = gt_table.get("page")
    gt_headers = gt_table.get("headers", [])

    best = None
    best_score = -1.0

    for ext_table in engine_tables:
        score = 0.0

        # Page match bonus
        ext_page = ext_table.get("page")
        if gt_page is not None and ext_page == gt_page:
            score += 1.0

        # Header overlap
        ext_headers = ext_table.get("headers", [])
        if gt_headers and ext_headers:
            score += _header_overlap(gt_headers, ext_headers)

        if score > best_score:
            best_score = score
            best = ext_table

    return best


def _header_overlap(gt_headers: list[str], ext_headers: list[str]) -> float:
    if not gt_headers:
        return 1.0 if not ext_headers else 0.0

    gt_norm = {_normalize(h) for h in gt_headers if h}
    ext_norm = {_normalize(h) for h in ext_headers if h}

    if not gt_norm:
        return 1.0

    intersection = gt_norm & ext_norm
    return len(intersection) / len(gt_norm)


def _score_sample_rows(
    gt_rows: list[list[str]],
    ext_table: dict[str, Any],
) -> float | None:
    if not gt_rows:
        return None

    ext_html = ext_table.get("html", "")
    if not ext_html:
        return 0.0

    # Extract cell texts from HTML
    ext_cells = re.findall(r"<td[^>]*>(.*?)</td>", ext_html, re.DOTALL)
    ext_text = " ".join(re.sub(r"<[^>]+>", "", c).strip() for c in ext_cells)

    gt_text = " ".join(" ".join(row) for row in gt_rows)

    return _fuzzy_score(gt_text, ext_text)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    if not args.benchmark_dir.exists():
        print(
            f"ERROR: Benchmark directory not found: {args.benchmark_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    gt_data = load_ground_truth(args.gt_csv_dir, args.ground_truth_dir)
    if not gt_data:
        print(
            f"WARN: No ground truth found in {args.gt_csv_dir} or {args.ground_truth_dir}"
        )
        print(
            "Add a metadata CSV under data/gt/ (file_name, carrier_inferred, scac_inferred, ...)."
        )
    else:
        csv_count = sum(1 for v in gt_data.values() if v.get("metadata"))
        print(f"Loaded {len(gt_data)} ground-truth entries ({csv_count} with metadata)")

    # Find engines
    if args.engines:
        engine_names = [e.strip() for e in args.engines.split(",")]
    else:
        engine_names = sorted(
            d.name
            for d in args.benchmark_dir.iterdir()
            if d.is_dir() and d.name != "summary.json"
        )

    print(f"Evaluating engines: {', '.join(engine_names)}")
    print(f"Ground truth files: {len(gt_data)}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)
    all_scorecards: list[dict[str, Any]] = []

    for engine_name in engine_names:
        print(f"=== {engine_name} ===")
        engine_results = find_engine_results(args.benchmark_dir, engine_name)

        if not engine_results:
            print(f"  No results found for {engine_name}")
            continue

        engine_scorecard: dict[str, Any] = {
            "engine": engine_name,
            "categories": {},
            "overall": {},
        }

        all_meta_scores: list[dict] = []
        all_table_scores: list[dict] = []

        for category, doc_results in engine_results.items():
            cat_scores: list[dict[str, Any]] = []

            for doc_stem, result_path in doc_results.items():
                result_data = json.loads(result_path.read_text(encoding="utf-8"))

                gt_key = sanitize_filename(doc_stem)
                gt = gt_data.get(gt_key, {})

                meta_score = score_metadata(result_data, gt)
                table_score = score_tables(result_data, gt)

                doc_score = {
                    "document": doc_stem,
                    "metadata": meta_score,
                    "tables": table_score,
                }
                cat_scores.append(doc_score)
                all_meta_scores.append(meta_score)
                all_table_scores.append(table_score)

            engine_scorecard["categories"][category] = cat_scores

        # Compute overall scores
        exact_rates = [
            s["exact_match_rate"]
            for s in all_meta_scores
            if s.get("exact_match_rate") is not None
        ]
        fuzzy_rates = [
            s["fuzzy_match_rate"]
            for s in all_meta_scores
            if s.get("fuzzy_match_rate") is not None
        ]
        header_accs = [
            s["avg_header_accuracy"]
            for s in all_table_scores
            if s.get("avg_header_accuracy") is not None
        ]
        row_accs = [
            s["avg_row_count_accuracy"]
            for s in all_table_scores
            if s.get("avg_row_count_accuracy") is not None
        ]

        engine_scorecard["overall"] = {
            "metadata_exact_match": (
                sum(exact_rates) / len(exact_rates) if exact_rates else None
            ),
            "metadata_fuzzy_match": (
                sum(fuzzy_rates) / len(fuzzy_rates) if fuzzy_rates else None
            ),
            "table_header_accuracy": (
                sum(header_accs) / len(header_accs) if header_accs else None
            ),
            "table_row_count_accuracy": (
                sum(row_accs) / len(row_accs) if row_accs else None
            ),
        }

        write_json(args.output / f"{engine_name}_scorecard.json", engine_scorecard)
        all_scorecards.append(engine_scorecard)

        overall = engine_scorecard["overall"]
        print(
            f"  Metadata exact: {_fmt(overall['metadata_exact_match'])}, "
            f"fuzzy: {_fmt(overall['metadata_fuzzy_match'])}"
        )
        print(
            f"  Tables header acc: {_fmt(overall['table_header_accuracy'])}, "
            f"row count acc: {_fmt(overall['table_row_count_accuracy'])}"
        )

        for category, cat_scores in engine_scorecard["categories"].items():
            for doc_score in cat_scores:
                meta = doc_score["metadata"]
                scorable = meta["total_fields"] - meta["missing_gt"]
                print(
                    f"    {category}/{doc_score['document']}: "
                    f"exact={meta['exact_matches']}/{scorable}, "
                    f"fuzzy={meta['fuzzy_matches']}/{scorable}, "
                    f"tables_gt={doc_score['tables']['gt_table_count']}, "
                    f"tables_ext={doc_score['tables']['extracted_table_count']}"
                )
                for field, detail in meta["fields"].items():
                    if detail.get("status") == "no_ground_truth":
                        continue
                    mark = "OK" if detail.get("exact_match") else "MISS"
                    print(
                        f"      {field}: gt={detail.get('gt')!r} "
                        f"ext={detail.get('extracted')!r} "
                        f"fuzzy={detail.get('fuzzy_score', 0):.2f} [{mark}]"
                    )
        print()

    # Write comparison summary
    comparison = _build_comparison(all_scorecards)
    write_json(args.output / "comparison.json", comparison)

    # Write detailed evaluation CSV
    csv_path = args.output / "evaluation_results.csv"
    _write_evaluation_csv(csv_path, all_scorecards, gt_data, engine_names)
    print(f"Evaluation CSV saved to: {csv_path}")

    print("Evaluation complete. Results saved to:", args.output)


def _fmt(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "N/A"


def _build_comparison(scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a side-by-side comparison table across engines."""
    rows: list[dict[str, Any]] = []

    for sc in scorecards:
        engine = sc.get("engine", "unknown")
        overall = sc.get("overall", {})
        rows.append(
            {
                "engine": engine,
                **overall,
            }
        )

    return {
        "comparison": rows,
        "best_metadata_exact": _best_engine(rows, "metadata_exact_match"),
        "best_metadata_fuzzy": _best_engine(rows, "metadata_fuzzy_match"),
        "best_table_headers": _best_engine(rows, "table_header_accuracy"),
        "best_table_rows": _best_engine(rows, "table_row_count_accuracy"),
    }


def _write_evaluation_csv(
    csv_path: Path,
    scorecards: list[dict[str, Any]],
    gt_data: dict[str, dict[str, Any]],
    engine_names: list[str],
) -> None:
    """Write per-engine CSVs and a combined comparison CSV."""
    META_FIELDS = ["carrier_name", "scac", "mode", "effective_date", "end_date"]

    # Collect all evaluated documents across engines
    doc_rows: dict[str, dict[str, Any]] = {}

    for sc in scorecards:
        engine = sc["engine"]
        for category, cat_scores in sc.get("categories", {}).items():
            for doc_score in cat_scores:
                doc_stem = doc_score["document"]
                if doc_stem not in doc_rows:
                    gt_key = sanitize_filename(doc_stem)
                    gt = gt_data.get(gt_key, {})
                    gt_meta = gt.get("metadata", {})
                    doc_rows[doc_stem] = {
                        "category": category,
                        "gt": {f: gt_meta.get(f) for f in META_FIELDS},
                        "engines": {},
                    }

                meta = doc_score["metadata"]
                fields = meta.get("fields", {})
                engine_info: dict[str, Any] = {}
                for f in META_FIELDS:
                    detail = fields.get(f, {})
                    engine_info[f"{f}_extracted"] = detail.get("extracted")
                    engine_info[f"{f}_source"] = detail.get("source_text")
                    engine_info[f"{f}_exact"] = detail.get("exact_match", False)
                    engine_info[f"{f}_fuzzy"] = detail.get("fuzzy_score", 0.0)

                scorable = meta["total_fields"] - meta["missing_gt"]
                engine_info["exact_rate"] = (
                    meta["exact_matches"] / scorable if scorable > 0 else None
                )
                engine_info["fuzzy_rate"] = (
                    meta["fuzzy_matches"] / scorable if scorable > 0 else None
                )
                engine_info["tables_extracted"] = doc_score["tables"][
                    "extracted_table_count"
                ]

                doc_rows[doc_stem]["engines"][engine] = engine_info

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _match_label(gt_val: str, exact: bool, fuzzy: float, ext_val: str) -> str:
        if not gt_val:
            return "NO_GT"
        if exact:
            return "EXACT"
        if fuzzy >= 0.8:
            return "FUZZY"
        if ext_val:
            return "PARTIAL"
        return "MISS"

    # --- Per-engine CSVs (GT vs Extracted side by side) ---
    for eng in engine_names:
        eng_csv = csv_path.parent / f"{eng}_evaluation.csv"
        headers = [
            "file_name",
            "category",
            "gt_carrier_name",
            f"{eng}_carrier_name",
            "carrier_match",
            "gt_scac",
            f"{eng}_scac",
            "scac_match",
            "gt_mode",
            f"{eng}_mode",
            "mode_match",
            "gt_effective_date",
            f"{eng}_effective_date",
            "effective_date_match",
            "gt_end_date",
            f"{eng}_end_date",
            "end_date_match",
            "exact_rate",
            "fuzzy_rate",
            "tables_extracted",
        ]

        with eng_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for doc_stem in sorted(doc_rows.keys()):
                info = doc_rows[doc_stem]
                eng_info = info["engines"].get(eng, {})
                row: list[Any] = [doc_stem, info["category"]]

                for fld in META_FIELDS:
                    gt_val = info["gt"].get(fld) or ""
                    ext_val = eng_info.get(f"{fld}_extracted") or ""
                    exact = eng_info.get(f"{fld}_exact", False)
                    fuzzy_sc = eng_info.get(f"{fld}_fuzzy", 0.0)
                    match = _match_label(gt_val, exact, fuzzy_sc, ext_val)
                    row.extend([gt_val, ext_val, match])

                exact_rate = eng_info.get("exact_rate")
                fuzzy_rate = eng_info.get("fuzzy_rate")
                row.append(f"{exact_rate:.1%}" if exact_rate is not None else "N/A")
                row.append(f"{fuzzy_rate:.1%}" if fuzzy_rate is not None else "N/A")
                row.append(eng_info.get("tables_extracted", 0))

                writer.writerow(row)

        print(f"  Per-engine CSV: {eng_csv}")

    # --- Combined comparison CSV (all engines side by side) ---
    combined_headers = ["file_name", "category"]
    for f in META_FIELDS:
        combined_headers.append(f"gt_{f}")
    for eng in engine_names:
        for f in META_FIELDS:
            combined_headers.append(f"{eng}_{f}")
            combined_headers.append(f"{eng}_{f}_match")
        combined_headers.append(f"{eng}_exact_rate")
        combined_headers.append(f"{eng}_fuzzy_rate")
        combined_headers.append(f"{eng}_tables")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(combined_headers)

        for doc_stem in sorted(doc_rows.keys()):
            info = doc_rows[doc_stem]
            row: list[Any] = [doc_stem, info["category"]]

            for fld in META_FIELDS:
                row.append(info["gt"].get(fld) or "")

            for eng in engine_names:
                eng_info = info["engines"].get(eng, {})
                for fld in META_FIELDS:
                    gt_val = info["gt"].get(fld) or ""
                    ext_val = eng_info.get(f"{fld}_extracted") or ""
                    exact = eng_info.get(f"{fld}_exact", False)
                    fuzzy_sc = eng_info.get(f"{fld}_fuzzy", 0.0)
                    row.append(ext_val)
                    row.append(_match_label(gt_val, exact, fuzzy_sc, ext_val))

                exact_rate = eng_info.get("exact_rate")
                fuzzy_rate = eng_info.get("fuzzy_rate")
                row.append(f"{exact_rate:.1%}" if exact_rate is not None else "N/A")
                row.append(f"{fuzzy_rate:.1%}" if fuzzy_rate is not None else "N/A")
                row.append(eng_info.get("tables_extracted", 0))

            writer.writerow(row)


def _best_engine(rows: list[dict[str, Any]], metric: str) -> str | None:
    valid = [(r["engine"], r.get(metric)) for r in rows if r.get(metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])[0]


if __name__ == "__main__":
    main()
