#!/usr/bin/env bash
set -euo pipefail

docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .
docker run --rm --network none \
  -v "$PWD:/workspace:ro" -w /workspace \
  plugin-dspy:3.4.1 python -m unittest discover -s python -v
echo "DSPY_DOCKER_TESTS=true" >> "$GITHUB_ENV"
