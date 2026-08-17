"""olmOCR 2 engine -- uses the olmocr CLI pipeline or Python API."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from engines import BaseEngine, DocumentResult, register_engine
from engines.olmocr.parse import parse_olmocr_output


@register_engine
class OlmOCREngine(BaseEngine):
    name = "olmocr"

    def __init__(
        self,
        *,
        model: str = "allenai/olmOCR-2-7B-1025",
        server_url: str | None = None,
    ) -> None:
        self._model = model
        self._server_url = server_url

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))

        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)

            cmd = [
                "python", "-m", "olmocr.pipeline",
                str(workspace_path),
                "--pdfs", str(pdf_path.resolve()),
                "--markdown",
                "--model", self._model,
            ]
            if self._server_url:
                cmd.extend(["--server", self._server_url])

            t0 = time.perf_counter()
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                result.errors.append(
                    {"page": 0, "stage": "pipeline", "error": "Timed out after 600s"}
                )
                result.timing = {"total_s": time.perf_counter() - t0}
                return result
            except subprocess.CalledProcessError as exc:
                result.errors.append(
                    {"page": 0, "stage": "pipeline", "error": exc.stderr or str(exc)}
                )
                result.timing = {"total_s": time.perf_counter() - t0}
                return result

            pipeline_s = time.perf_counter() - t0

            try:
                regions, tables = parse_olmocr_output(workspace_path)
                result.regions = regions
                result.tables = tables
            except Exception as exc:
                result.errors.append(
                    {"page": 0, "stage": "parse", "error": str(exc)}
                )

            result.timing = {"total_s": pipeline_s}

        return result
