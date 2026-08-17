# API Cost Tracking

Approximate list prices used by the benchmark (USD). Override via env vars when your SKU differs.

| Engine | Provider | Default rate | Env override | What we bill |
|--------|----------|--------------|--------------|--------------|
| `azure` | Azure Document Intelligence `prebuilt-layout` | **$10 / 1K pages** | `AZURE_DI_USD_PER_1K_PAGES` | Every PDF page sent to ADI |
| `azure-hybrid` | Same ADI model, table crops only | **$10 / 1K pages** | `AZURE_DI_USD_PER_1K_PAGES` | **1 page per table crop** sent to ADI (pages with no tables = $0 ADI) |
| `mistral` | Mistral OCR 4 | **$4 / 1K pages** ($2 batch) | `MISTRAL_OCR_USD_PER_1K_PAGES` | Every PDF page returned by the OCR response |
| Local engines (`pp-structure`, `paddle-vl`, `olmocr`, `monkeyocr`) | — | $0 | — | No API cost |

Sources (verify against your contract):
- [Azure Document Intelligence pricing](https://azure.microsoft.com/pricing/details/ai-document-intelligence/)
- [Mistral OCR 4](https://docs.mistral.ai/models/model-cards/ocr-4-0) — listed at $4 / 1K pages

## Hybrid Azure flow (`azure-hybrid`)

```
PDF page
  │
  ├─► PP-DocLayoutV3  ── detect layout boxes
  │         │
  │         ├─ table boxes ── crop ──► Azure DI prebuilt-layout (billed)
  │         │
  │         └─ non-table area
  │
  └─► PP-OCRv6 on full page ── drop OCR regions overlapping tables
```

- **Layout**: `LayoutDetection(model_name="PP-DocLayoutV3")`
- **Text**: local `PaddleOCR(ocr_version="PP-OCRv6")`
- **Tables**: only cropped table images go to Azure

This is the mode to compare against full-document `azure` for cost vs table quality.

## Where cost appears in outputs

Each `result.json` includes a `cost` object:

```json
{
  "provider": "azure-document-intelligence",
  "unit": "page",
  "units_billed": 3,
  "usd_per_1k_units": 10.0,
  "usd_total": 0.03,
  "notes": "...",
  "breakdown": { "azure_api_calls": 3, "tables_detected": 3 }
}
```

Benchmark summaries roll up `total_cost_usd` and `total_units_billed` per engine.
