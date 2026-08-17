"""MonkeyOCR v2 engine -- talks to a running vLLM server via the MonkeyOCR parse API."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from engines import BaseEngine, DocumentResult, register_engine
from engines.monkeyocr.parse import parse_monkeyocr_output


@register_engine
class MonkeyOCREngine(BaseEngine):
    name = "monkeyocr"

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:8888",
        model_path: str | None = None,
    ) -> None:
        self._server_url = server_url
        self._model_path = model_path

    def process(self, pdf_path: Path) -> DocumentResult:
        result = DocumentResult(engine=self.name, pdf_path=str(pdf_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            cmd = [
                "python", "parse.py",
                "-i", str(pdf_path.resolve()),
                "-o", str(output_dir),
                "-s", self._server_url,
                "--draw-layout",
            ]
            if self._model_path:
                cmd.extend(["-m", self._model_path])

            t0 = time.perf_counter()
            try:
                # MonkeyOCR's parse.py expects to run from its parsing/ dir.
                # If monkeyocr is installed as a package, try the module approach;
                # otherwise fall back to the HTTP API.
                self._run_via_api(pdf_path, output_dir, result)
            except Exception:
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                except Exception as exc:
                    result.errors.append(
                        {"page": 0, "stage": "pipeline", "error": str(exc)}
                    )
                    result.timing = {"total_s": time.perf_counter() - t0}
                    return result

            pipeline_s = time.perf_counter() - t0

            if not result.regions and not result.errors:
                try:
                    regions, tables = parse_monkeyocr_output(output_dir)
                    result.regions = regions
                    result.tables = tables
                except Exception as exc:
                    result.errors.append(
                        {"page": 0, "stage": "parse", "error": str(exc)}
                    )

            result.timing = {"total_s": pipeline_s}

        return result

    def _run_via_api(
        self, pdf_path: Path, output_dir: Path, result: DocumentResult
    ) -> None:
        """Call MonkeyOCR via its FastAPI endpoint if available."""
        import requests

        api_url = self._server_url.rstrip("/")

        # MonkeyOCR FastAPI typically exposes /parse endpoint
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{api_url}/parse",
                files={"file": (pdf_path.name, f, "application/pdf")},
                timeout=300,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"MonkeyOCR API returned {resp.status_code}: {resp.text}")

        data = resp.json()

        # Write response for parse_monkeyocr_output to pick up
        out_file = output_dir / f"{pdf_path.stem}.json"
        out_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        regions, tables = parse_monkeyocr_output(output_dir)
        result.regions = regions
        result.tables = tables
