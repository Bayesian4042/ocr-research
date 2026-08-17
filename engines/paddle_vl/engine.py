"""PaddleOCR-VL 1.6 engine -- vision-language model for document parsing."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from engines import BaseEngine, DocumentResult, register_engine
from engines.paddle_vl.parse import parse_vl_result


@register_engine
class PaddleVLEngine(BaseEngine):
    name = "paddle-vl"

    def __init__(self, *, use_orientation_classify: bool = True) -> None:
        from paddleocr import PaddleOCRVL

        self._pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            use_doc_orientation_classify=use_orientation_classify,
        )

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))

        t0 = time.perf_counter()
        try:
            raw_output = self._pipeline.predict(str(pdf_path))
        except Exception as exc:
            result.errors.append({"page": 0, "stage": "predict", "error": str(exc)})
            result.timing = {"total_s": time.perf_counter() - t0}
            return result

        predict_s = time.perf_counter() - t0

        with tempfile.TemporaryDirectory() as tmpdir:
            for page_idx, page_result in enumerate(raw_output):
                page_num = page_idx + 1
                try:
                    json_path = Path(tmpdir) / f"page_{page_num}.json"
                    page_result.save_to_json(save_path=str(json_path.parent))

                    saved_files = list(Path(tmpdir).glob("*.json"))
                    if saved_files:
                        latest = max(saved_files, key=lambda p: p.stat().st_mtime)
                        page_data = json.loads(latest.read_text(encoding="utf-8"))
                    else:
                        page_data = {}

                    regions, tables = parse_vl_result(page_data, page_num)
                    result.regions.extend(regions)
                    result.tables.extend(tables)
                except Exception as exc:
                    result.errors.append(
                        {"page": page_num, "stage": "parse", "error": str(exc)}
                    )

        result.timing = {"total_s": predict_s}
        return result
