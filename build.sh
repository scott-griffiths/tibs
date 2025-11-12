#!/bin/bash
set -e  # Exit on any error

# Function to log steps
log() {
    echo "==> $1"
}

if ! command -v uv &> /dev/null; then
    log "uv not found. Please install uv first."
    exit 1
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    log "Creating uv virtual environment"
    uv venv
fi

log "Activating virtual environment"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install dependencies including dev dependencies
log "Installing dependencies"
uv pip install ".[dev]"

# Build and install the entire package
log "Building and installing package"
uvx maturin develop

# Run tests
log "Running tests"
uv run pytest --benchmark-skip