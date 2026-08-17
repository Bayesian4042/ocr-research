#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="$BASE/data/Archive"
DEST="$BASE/data/test-dataset"

if [ ! -d "$ARCHIVE" ]; then
  ARCHIVE="$BASE/data/Archieve"
fi

if [ ! -d "$ARCHIVE" ]; then
  echo "ERROR: Neither data/Archive nor data/Archieve found." >&2
  exit 1
fi

mkdir -p "$DEST/orientation" "$DEST/scanned" "$DEST/complex-tables" "$DEST/ground-truth"

orientation_files=(
  "Dugan.Ecolab 2023 LTL RFP Primary Rate Card Executed.pdf"
  "US Special 3-3 Zip 2023 LTL RFP Primary Rate Card Executed.pdf"
  "Midland Transport_MDLD_LTL_AMD_Rates_9.20.2023.pdf"
  "Old Dominion_ODFL_MSA_8.3.2023.pdf"
  "Skyline_SKYT_TL_AMD_McDonough OB_10.13.2023.pdf"
  "Dedicated Logistics_DCLH_TL_AMD_Backup Rates_11.29.2022.pdf"
)

scanned_files=(
  "Dedicated Delivery Professionals 2023 LTL RFP Primary Rate Card Executed.pdf"
  "Armour Transportation Systems 2022-2023 LTL Contract Executed.pdf"
  "Dugan.Ecolab 2023 MSA signed 4.7.2023 (2).pdf"
  "Dupre_DUPR_MSA_3.21.2017 (1).pdf"
  "AIT_TL_MSA_6.2.2015.pdf"
  "Posse - Ecolab MSA.pdf"
)

complex_table_files=(
  "Saia.Ecolab 2023 LTL RFP Primary Rate Card Executed.pdf"
  "Pitt Ohio 2023 LTL RFP Primary Rate Card Executed.pdf"
  "Dohrn.Ecolab 2023 LTL RFP Primary Rate Card Executed.pdf"
  "ABF_ABFS_LTL_AMD_Secondary Award Rate Card_5.1.2023.pdf"
  "R&L_RNLO_LTL_AMD_2023 RFP Secondary_04.15.2023.pdf"
  "Oceanex_OCXI_FCL_AMD_Rate Sheet_9.22.2023.pdf"
)

moved=0
missing=0

move_files() {
  local category="$1"
  shift
  local files=("$@")
  for f in "${files[@]}"; do
    src="$ARCHIVE/$f"
    if [ -f "$src" ]; then
      cp "$src" "$DEST/$category/$f"
      echo "  OK  $category/$f"
      moved=$((moved + 1))
    else
      echo "  MISSING  $src"
      missing=$((missing + 1))
    fi
  done
}

echo "Copying PDFs from $ARCHIVE -> $DEST"
echo

echo "=== Orientation / Nasty Cases ==="
move_files "orientation" "${orientation_files[@]}"

echo
echo "=== True Scanned / Image-Only ==="
move_files "scanned" "${scanned_files[@]}"

echo
echo "=== Complex Table Baseline ==="
move_files "complex-tables" "${complex_table_files[@]}"

echo
echo "Done. $moved copied, $missing missing."
