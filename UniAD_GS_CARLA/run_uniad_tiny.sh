#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT="${UNIAD_TINY_CHECKPOINT:-${SCRIPT_DIR}/checkpoints/uniad_tiny_b2d.pth}"
CONFIG="${UNIAD_TINY_CONFIG:-${BENCH2DRIVE_ZOO_ROOT:-${REPO_ROOT}/Bench2DriveZoo}/adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/uniad_gs_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing UniAD-Tiny B2D checkpoint: ${CHECKPOINT}" >&2
  echo "Download uniad_tiny_b2d.pth into ${SCRIPT_DIR}/checkpoints/" >&2
  echo "or set UNIAD_TINY_CHECKPOINT=/absolute/path/uniad_tiny_b2d.pth" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing UniAD-Tiny config: ${CONFIG}" >&2
  echo "Set BENCH2DRIVE_ZOO_ROOT or UNIAD_TINY_CONFIG to the external Bench2DriveZoo checkout." >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec python "${SCRIPT_DIR}/run_uniad_agent.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  "$@"
