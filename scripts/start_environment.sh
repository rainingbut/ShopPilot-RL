#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/environments/ShopSimulator/.venv-shopsim"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "ShopSimulator is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
exec "$ROOT/environments/ShopSimulator/shop_env/start.sh"
