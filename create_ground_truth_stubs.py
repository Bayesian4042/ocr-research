"""Generate empty ground-truth stub JSON files for every PDF in the test dataset."""

import json
import re
from pathlib import Path

DATASET_DIR = Path("data/test-dataset")
GT_DIR = DATASET_DIR / "ground-truth"

CATEGORIES = ("orientation", "scanned", "complex-tables")


def sanitize_filename(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)


def make_stub(pdf_name: str, category: str) -> dict:
    return {
        "source_pdf": pdf_name,
        "category": category,
        "metadata": {
            "carrier_name": None,
            "contract_type": None,
            "effective_date": None,
            "page_count": None,
        },
        "tables": [
            {
                "page": None,
                "description": None,
                "headers": [],
                "row_count": None,
                "sample_rows": [],
            }
        ],
    }


def main() -> None:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    created = 0

    for category in CATEGORIES:
        cat_dir = DATASET_DIR / category
        if not cat_dir.exists():
            print(f"  SKIP  {cat_dir} (does not exist yet)")
            continue

        for pdf in sorted(cat_dir.glob("*.pdf")):
            stub = make_stub(pdf.name, category)
            out_path = GT_DIR / f"{sanitize_filename(pdf.name)}.json"
            out_path.write_text(
                json.dumps(stub, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  OK  {out_path.name}")
            created += 1

    if created == 0:
        print(
            "\nNo PDFs found. Run organize_dataset.sh first, then re-run this script."
        )
        print("Creating stubs from manifest.json instead...\n")
        manifest_path = DATASET_DIR / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for category, info in manifest.get("categories", {}).items():
                for pdf_name in info.get("files", []):
                    stub = make_stub(pdf_name, category)
                    out_path = GT_DIR / f"{sanitize_filename(pdf_name)}.json"
                    out_path.write_text(
                        json.dumps(stub, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    print(f"  OK  {out_path.name}")
                    created += 1

    print(f"\nDone. {created} ground-truth stubs created in {GT_DIR}")


if __name__ == "__main__":
    main()
