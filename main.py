import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from paddleocr import PaddleOCR, TableRecognitionPipelineV2


DEFAULT_CORPUS_DIR = Path("data/Archieve")
FALLBACK_CORPUS_DIR = Path("data/Archive")
OUTPUTS_DIR = Path("outputs")
ENGINE_NAME = "ppocrv6"
TABLE_ENGINE_NAME = "table_recognition_v2"
PASS_NUMBER = 1
CONFIDENCE_THRESHOLD = 0.80
PDF_DPI = 300


@dataclass
class BenchmarkStats:
    documents: int = 0
    pages: int = 0
    regions: int = 0
    confidence_sum: float = 0.0
    low_confidence_regions: int = 0
    failed_pages: int = 0
    render_seconds: float = 0.0
    ocr_seconds: float = 0.0
    table_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PP-OCRv6 baseline OCR pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs")
    return parser.parse_args()


def resolve_corpus_dir() -> Path:
    if DEFAULT_CORPUS_DIR.exists():
        return DEFAULT_CORPUS_DIR
    if FALLBACK_CORPUS_DIR.exists():
        return FALLBACK_CORPUS_DIR
    return DEFAULT_CORPUS_DIR


def discover_pdfs(corpus_dir: Path, limit: int | None) -> list[Path]:
    pdfs = [
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    pdfs.sort()
    if limit is not None:
        return pdfs[:limit]
    return pdfs


def pixmap_to_numpy(pix: pymupdf.Pixmap) -> np.ndarray:
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def to_bbox(points: Any) -> list[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def get_result_field(result: Any, field_name: str) -> Any:
    if isinstance(result, dict):
        return result.get(field_name)
    return getattr(result, field_name, None)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_ocr_result(raw_ocr: Any, page_number: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if not raw_ocr:
        return regions

    # PaddleOCR 2.x style: [[[points, (text, confidence)], ...]]
    first_item = raw_ocr[0] if isinstance(raw_ocr, list) and raw_ocr else None
    if isinstance(first_item, list):
        for entry in first_item:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue

            points = entry[0]
            text_conf = entry[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue

            text = text_conf[0]
            confidence = float(text_conf[1])
            if not isinstance(text, str):
                text = str(text)

            regions.append(
                {
                    "page": page_number,
                    "text": text,
                    "bbox": to_bbox(points),
                    "confidence": confidence,
                    "engine": ENGINE_NAME,
                    "pass": PASS_NUMBER,
                }
            )
        return regions

    # PaddleOCR 3.x style: [OCRResult(...)] with dt_polys/rec_texts/rec_scores
    page_results = raw_ocr if isinstance(raw_ocr, list) else [raw_ocr]
    for page_result in page_results:
        dt_polys = get_result_field(page_result, "dt_polys")
        rec_texts = get_result_field(page_result, "rec_texts")
        rec_scores = get_result_field(page_result, "rec_scores")

        if dt_polys is None or rec_texts is None:
            continue

        n = min(len(dt_polys), len(rec_texts))
        if rec_scores is not None:
            n = min(n, len(rec_scores))

        for idx in range(n):
            points = dt_polys[idx]
            if hasattr(points, "tolist"):
                points = points.tolist()

            text = rec_texts[idx]
            if not isinstance(text, str):
                text = str(text)

            confidence = float(rec_scores[idx]) if rec_scores is not None else 0.0

            regions.append(
                {
                    "page": page_number,
                    "text": text,
                    "bbox": to_bbox(points),
                    "confidence": confidence,
                    "engine": ENGINE_NAME,
                    "pass": PASS_NUMBER,
                }
            )

    return regions


def parse_table_result(raw_table: Any, page_number: int) -> list[dict[str, Any]]:
    if not raw_table:
        return []

    table_results = raw_table if isinstance(raw_table, list) else [raw_table]
    parsed: list[dict[str, Any]] = []

    for table_result in table_results:
        if hasattr(table_result, "to_dict"):
            result_data = table_result.to_dict()
        elif isinstance(table_result, dict):
            result_data = table_result
        else:
            result_data = {
                key: getattr(table_result, key)
                for key in dir(table_result)
                if not key.startswith("_")
                and not callable(getattr(table_result, key, None))
            }

        parsed.append(
            {
                "page": page_number,
                "engine": TABLE_ENGINE_NAME,
                "pass": PASS_NUMBER,
                "result": to_jsonable(result_data),
            }
        )

    return parsed


def build_output_dir(corpus_dir: Path, pdf_path: Path) -> Path:
    relative = pdf_path.relative_to(corpus_dir).with_suffix("")
    return OUTPUTS_DIR / relative


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def process_pdf(
    ocr_engine: PaddleOCR,
    table_engine: TableRecognitionPipelineV2,
    corpus_dir: Path,
    pdf_path: Path,
    stats: BenchmarkStats,
) -> None:
    result_regions: list[dict[str, Any]] = []
    result_tables: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            stats.pages += 1

            try:
                render_start = time.perf_counter()
                page = document.load_page(page_index)
                matrix = pymupdf.Matrix(PDF_DPI / 72.0, PDF_DPI / 72.0)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = pixmap_to_numpy(pix)
                stats.render_seconds += time.perf_counter() - render_start
            except Exception as exc:  # noqa: BLE001
                stats.failed_pages += 1
                failed_pages.append(
                    {"page": page_number, "stage": "render", "error": str(exc)}
                )
                continue

            try:
                ocr_start = time.perf_counter()
                raw = ocr_engine.predict(image)
                stats.ocr_seconds += time.perf_counter() - ocr_start
            except Exception as exc:  # noqa: BLE001
                stats.failed_pages += 1
                failed_pages.append(
                    {"page": page_number, "stage": "ocr", "error": str(exc)}
                )
                continue

            page_regions = parse_ocr_result(raw, page_number)
            result_regions.extend(page_regions)

            for region in page_regions:
                stats.regions += 1
                stats.confidence_sum += region["confidence"]
                if region["confidence"] < CONFIDENCE_THRESHOLD:
                    stats.low_confidence_regions += 1

            try:
                table_start = time.perf_counter()
                table_raw = table_engine.predict(image)
                stats.table_seconds += time.perf_counter() - table_start
                result_tables.extend(parse_table_result(table_raw, page_number))
            except Exception as exc:  # noqa: BLE001
                stats.failed_pages += 1
                failed_pages.append(
                    {"page": page_number, "stage": "table", "error": str(exc)}
                )
                continue

    output_dir = build_output_dir(corpus_dir, pdf_path)
    write_json(
        output_dir / "result.json",
        {
            "document": str(pdf_path.relative_to(corpus_dir)),
            "regions": result_regions,
            "tables": result_tables,
            "failed_pages": failed_pages,
        },
    )


def build_benchmark(stats: BenchmarkStats) -> dict[str, Any]:
    total_processing_seconds = stats.render_seconds + stats.ocr_seconds
    average_confidence = stats.confidence_sum / stats.regions if stats.regions else 0.0
    pages_per_hour = (
        (stats.pages / total_processing_seconds) * 3600
        if total_processing_seconds > 0
        else 0.0
    )

    return {
        "documents": stats.documents,
        "pages": stats.pages,
        "regions": stats.regions,
        "average confidence": average_confidence,
        "regions with confidence < 0.80": stats.low_confidence_regions,
        "failed pages": stats.failed_pages,
        "render seconds": stats.render_seconds,
        "OCR seconds": stats.ocr_seconds,
        "table seconds": stats.table_seconds,
        "pages/hour": pages_per_hour,
    }


def create_ocr_engine() -> PaddleOCR:
    return PaddleOCR(ocr_version="PP-OCRv6")


def create_table_engine() -> TableRecognitionPipelineV2:
    try:
        return TableRecognitionPipelineV2()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TableRecognitionPipelineV2 dependencies are missing. "
            'Install with: uv add "paddlex[ocr]" && uv sync'
        ) from exc


def main() -> None:
    args = parse_args()
    corpus_dir = resolve_corpus_dir()

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    pdf_paths = discover_pdfs(corpus_dir, args.limit)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    stats = BenchmarkStats(documents=len(pdf_paths))
    ocr_engine = create_ocr_engine()
    table_engine = create_table_engine()

    for pdf_path in pdf_paths:
        process_pdf(ocr_engine, table_engine, corpus_dir, pdf_path, stats)

    write_json(OUTPUTS_DIR / "benchmark.json", build_benchmark(stats))


if __name__ == "__main__":
    main()
