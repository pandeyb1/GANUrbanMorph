#!/usr/bin/env bash
set -euo pipefail

cd /Users/9oy/Documents/Projects/IM3/EvaluationP

BASE="Report/GIScRSSubmission/Revision/codev3/Stratified_Archive"
GIS_PYTHON="/opt/miniconda3/envs/gis/bin/python"
CACHE_DIR="${BASE}/state_cache"
STATE_MANIFEST_DIR="${CACHE_DIR}/_per_state_manifests"
STATE_SENTINEL_DIR="${CACHE_DIR}/.complete"
MA_INPUT_DIR="/Users/9oy/Documents/Data/IM3/Buildings/MAv1"

export CONDA_DEFAULT_ENV="gis"
export CONDA_EXE="/opt/miniconda3/bin/conda"
export CONDA_PREFIX="/opt/miniconda3/envs/gis"
export CONDA_PREFIX_1="/opt/miniconda3"
export CONDA_PROMPT_MODIFIER="(gis) "
export CONDA_PYTHON_EXE="/opt/miniconda3/bin/python"
export CONDA_ROOT="/opt/miniconda3"
export CONDA_SHLVL="2"
export GDAL_DATA="/opt/miniconda3/envs/gis/share/gdal"
export GDAL_DRIVER_PATH="/opt/miniconda3/envs/gis/lib/gdalplugins"
export PROJ_DATA="/opt/miniconda3/envs/gis/share/proj"
export PROJ_NETWORK="ON"
export PATH="/opt/miniconda3/envs/gis/bin:${PATH}"

mkdir -p "${CACHE_DIR}" "${STATE_MANIFEST_DIR}" "${STATE_SENTINEL_DIR}"
shopt -s nullglob

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "[$(timestamp)] Starting resumable state-cache build under ${CACHE_DIR}"

for csv_path in "${MA_INPUT_DIR}"/*.csv; do
  state_code="$(basename "${csv_path}" .csv)"
  cache_path="${CACHE_DIR}/${state_code}.gpkg"
  sentinel_path="${STATE_SENTINEL_DIR}/${state_code}.done"

  if [[ -s "${cache_path}" ]]; then
    if [[ ! -f "${sentinel_path}" ]]; then
      touch "${sentinel_path}"
    fi
    echo "[$(timestamp)] Skipping ${state_code}; cache output already exists."
    continue
  fi

  rm -f "${cache_path}" "${sentinel_path}"
  rm -f "${CACHE_DIR}/state_cache_manifest.csv" "${CACHE_DIR}/state_cache_summary.json"

  echo "[$(timestamp)] Building cache for ${state_code}"
  "${GIS_PYTHON}" -u "${BASE}/prepare_ma_state_polygon_cache.py" \
    --states "${state_code}" \
    --state-cache-dir "${CACHE_DIR}"

  cp "${CACHE_DIR}/state_cache_manifest.csv" \
    "${STATE_MANIFEST_DIR}/${state_code}_state_cache_manifest.csv"
  cp "${CACHE_DIR}/state_cache_summary.json" \
    "${STATE_MANIFEST_DIR}/${state_code}_state_cache_summary.json"
  touch "${sentinel_path}"
done

echo "[$(timestamp)] Rebuilding combined state-cache manifest files"
"${GIS_PYTHON}" - <<'PY'
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

base = Path("/Users/9oy/Documents/Projects/IM3/EvaluationP/Report/GIScRSSubmission/Revision/codev3/Stratified_Archive/state_cache")
manifest_dir = base / "_per_state_manifests"
manifest_files = sorted(manifest_dir.glob("*_state_cache_manifest.csv"))
summary_files = sorted(manifest_dir.glob("*_state_cache_summary.json"))

frames = [pd.read_csv(path) for path in manifest_files]
if frames:
    manifest = pd.concat(frames, ignore_index=True).sort_values("state_code").reset_index(drop=True)
else:
    manifest = pd.DataFrame(
        columns=[
            "state_code",
            "source_csv",
            "cache_path",
            "source_row_count",
            "cached_feature_count",
            "dropped_malformed_empty_features",
            "invalid_before_repair",
            "invalid_after_repair",
            "area2d_rel_err_mean",
            "area2d_rel_err_median",
            "area2d_rel_err_p95",
        ]
    )
manifest.to_csv(base / "state_cache_manifest.csv", index=False)

summary = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "state_cache_dir": str(base),
    "manifest_file_count": len(manifest_files),
    "states": {},
}
for path in summary_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary["input_dir"] = payload.get("input_dir")
    summary["nlcd_2015"] = payload.get("nlcd_2015")
    summary["target_proj4"] = payload.get("target_proj4")
    summary["states"].update(payload.get("states", {}))

with (base / "state_cache_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
PY

"${GIS_PYTHON}" -u "${BASE}/build_stratified_test_archive.py" \
  --stage all \
  --state-cache-dir "${CACHE_DIR}" \
  --output-dir "${BASE}/output_test"
