#!/bin/bash
set -euo pipefail

echo "Building and starting Hammerthrow Analysis (Docker)..."
docker-compose up --build -d

echo
echo "Started. Open your browser at: http://localhost:5000"
echo "To stop the app: docker-compose down"
