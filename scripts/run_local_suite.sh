#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible local-model entry.
# Usage:
#   bash scripts/run_local_suite.sh negative 100
#   bash scripts/run_local_suite.sh negative full

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_KIND="${1:-negative}"
SIZE="${2:-100}"
shift $(( $# >= 2 ? 2 : $# )) || true

exec "$PROJECT_ROOT/run_models.sh" small "$SIZE" --dataset-kind "$DATASET_KIND" "$@"
