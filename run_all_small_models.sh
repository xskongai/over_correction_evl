#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias. Model execution only; no automatic summarization.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_ROOT/run_models.sh" small "$@"
