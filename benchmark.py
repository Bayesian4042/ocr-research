"""OCR Benchmark Runner -- iterates engines x documents, saves per-engine results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from engines import (
    BaseEngine,
    DocumentResult,
    available_engines,
    get_engine_class,
    import_all_engines,
)

DEFAULT_DATASET = Path("data/test-dataset")
DEFAULT_OUTPUT = Path("outputs/benchmark")
CATEGORIES = ("orientation", "scanned", "complex-tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR benchmark across engines and test documents"
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help="Comma-separated engine names (default: all available)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to test-dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for results",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Run only a specific category (orientation, scanned, complex-tables)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDFs per category",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Seconds to wait between documents (helps with API rate limits)",
    )
    return parser.parse_args()


def discover_pdfs(
    dataset_dir: Path, category: str | None, limit: int | None
) -> dict[str, list[Path]]:
    """Return {category: [pdf_paths]} for the test dataset."""
    categories = [category] if category else list(CATEGORIES)
    result: dict[str, list[Path]] = {}

    for cat in categories:
        cat_dir = dataset_dir / cat
        if not cat_dir.exists():
            print(f"  WARN: category directory not found: {cat_dir}")
            continue

        pdfs = sorted(
            p for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
        )
        if limit is not None:
            pdfs = pdfs[:limit]
        if pdfs:
            result[cat] = pdfs

    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_engine(
    engine: BaseEngine,
    pdfs_by_category: dict[str, list[Path]],
    output_dir: Path,
    delay: float = 0,
) -> dict[str, Any]:
    """Run a single engine across all documents. Returns engine-level summary."""
    engine_dir = output_dir / engine.name
    engine_summary: dict[str, Any] = {
        "engine": engine.name,
        "categories": {},
        "total_documents": 0,
        "total_pages": 0,
        "total_regions": 0,
        "total_tables": 0,
        "total_errors": 0,
        "total_time_s": 0.0,
        "total_cost_usd": 0.0,
        "total_units_billed": 0.0,
        "cost_provider": None,
    }

    doc_global_idx = 0
    for category, pdf_paths in pdfs_by_category.items():
        cat_results: list[dict[str, Any]] = []

        for pdf_path in pdf_paths:
            if delay > 0 and doc_global_idx > 0:
                time.sleep(delay)
            doc_global_idx += 1

            doc_stem = pdf_path.stem
            print(f"    [{engine.name}] {category}/{doc_stem}")

            try:
                doc_result = engine.process(pdf_path)
            except Exception as exc:
                doc_result = DocumentResult(
                    engine=engine.name,
                    pdf_path=str(pdf_path),
                    errors=[{"page": 0, "stage": "engine", "error": str(exc)}],
                )

            result_dict = doc_result.to_dict()
            result_path = engine_dir / category / doc_stem / "result.json"
            write_json(result_path, result_dict)

            cost_dict = doc_result.cost.to_dict() if doc_result.cost else None
            cat_results.append(
                {
                    "document": doc_stem,
                    "regions": len(doc_result.regions),
                    "tables": len(doc_result.tables),
                    "errors": len(doc_result.errors),
                    "timing": doc_result.timing,
                    "cost": cost_dict,
                }
            )

            engine_summary["total_documents"] += 1
            engine_summary["total_regions"] += len(doc_result.regions)
            engine_summary["total_tables"] += len(doc_result.tables)
            engine_summary["total_errors"] += len(doc_result.errors)
            engine_summary["total_time_s"] += sum(doc_result.timing.values())
            if doc_result.cost:
                engine_summary["total_cost_usd"] += doc_result.cost.usd_total
                engine_summary["total_units_billed"] += doc_result.cost.units_billed
                engine_summary["cost_provider"] = doc_result.cost.provider

        engine_summary["categories"][category] = {
            "documents": len(cat_results),
            "results": cat_results,
        }

    engine_summary["total_cost_usd"] = round(engine_summary["total_cost_usd"], 6)
    return engine_summary


def main() -> None:
    import_all_engines()
    args = parse_args()

    if not args.dataset.exists():
        print(f"ERROR: Dataset directory not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Determine which engines to run
    if args.engines:
        engine_names = [e.strip() for e in args.engines.split(",")]
    else:
        engine_names = available_engines()

    if not engine_names:
        print("ERROR: No engines available. Check your dependencies.", file=sys.stderr)
        sys.exit(1)

    print(f"Engines: {', '.join(engine_names)}")
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    print()

    pdfs_by_category = discover_pdfs(args.dataset, args.category, args.limit)
    total_pdfs = sum(len(v) for v in pdfs_by_category.values())
    print(f"Found {total_pdfs} PDFs across {len(pdfs_by_category)} categories")
    print()

    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []

    for engine_name in engine_names:
        print(f"=== Engine: {engine_name} ===")
        try:
            engine_cls = get_engine_class(engine_name)
            engine = engine_cls()
        except Exception as exc:
            print(f"  SKIP: Failed to initialize {engine_name}: {exc}")
            all_summaries.append({"engine": engine_name, "error": str(exc)})
            continue

        t0 = time.perf_counter()
        summary = run_engine(engine, pdfs_by_category, args.output, delay=args.delay)
        summary["wall_time_s"] = time.perf_counter() - t0
        all_summaries.append(summary)

        write_json(args.output / engine_name / "summary.json", summary)
        cost_str = ""
        if summary.get("total_cost_usd", 0) > 0:
            cost_str = f", ${summary['total_cost_usd']:.4f} est. cost"
        print(
            f"  Done: {summary['total_documents']} docs, "
            f"{summary['total_regions']} regions, "
            f"{summary['total_tables']} tables, "
            f"{summary['total_errors']} errors, "
            f"{summary['wall_time_s']:.1f}s wall time"
            f"{cost_str}"
        )
        print()

    # Write combined summary
    write_json(
        args.output / "summary.json",
        {
            "engines": all_summaries,
            "dataset": str(args.dataset),
            "total_documents_per_engine": total_pdfs,
        },
    )

    print("Benchmark complete. Results saved to:", args.output)


if __name__ == "__main__":
    main()
